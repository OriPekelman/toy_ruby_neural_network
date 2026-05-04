require "net/http"
require "openssl"
require "uri"
require "fileutils"
require "digest"

# Tiny HuggingFace dataset file fetcher.
#
# Why not just use the durable_huggingface_hub gem?
#   The gem (0.2.0) doesn't enable Faraday's follow_redirects middleware,
#   so any HEAD/GET against `huggingface.co/datasets/<repo>/resolve/...`
#   fails on the 307 to `/api/resolve-cache/...`. Even if we patch in
#   the middleware, the gem's `download_to_blob` streams via Faraday's
#   `on_data` callback while opening the temp file in `"ab"` (append)
#   mode — the 307 response body ("Temporary Redirect…") gets written
#   first, then the real content gets appended. Filed upstream:
#     https://github.com/durableprogramming/durable-huggingface-hub-ruby/issues/1
#
# In the meantime: stdlib `Net::HTTP.get_response` follows redirects on
# its own once we walk them ourselves, and we cache files on disk under
# the same `~/.cache/huggingface/datasets-toy-rnn/` shape so re-runs are
# free.
module DatasetLoader
  module_function

  HF_BASE   = "https://huggingface.co".freeze
  CACHE_DIR = File.expand_path("~/.cache/huggingface/datasets-toy-rnn").freeze

  # Download (or load from cache) a file from a HuggingFace dataset repo.
  # Returns the local path.
  def fetch(repo_id, filename, repo_type: "dataset", revision: "main")
    type_seg = repo_type == "model" ? "" : "#{repo_type}s/"
    url      = "#{HF_BASE}/#{type_seg}#{repo_id}/resolve/#{revision}/#{filename}"
    key      = Digest::SHA256.hexdigest("#{repo_type}|#{repo_id}|#{revision}|#{filename}")[0, 16]
    safename = filename.tr("/", "_")
    path     = File.join(CACHE_DIR, "#{key}_#{safename}")

    return path if File.file?(path) && File.size(path) > 0

    FileUtils.mkdir_p(CACHE_DIR)
    download(url, path)
    path
  end

  # Read a HF dataset file and return chomped, non-empty lines.
  def lines(repo_id, filename, **opts)
    File.readlines(fetch(repo_id, filename, **opts), chomp: true)
        .reject { |l| l.strip.empty? }
  end

  # First `n` non-empty lines — handy for slicing big corpora.
  def head(repo_id, filename, n, **opts)
    lines(repo_id, filename, **opts).first(n)
  end

  # ------------------------------------------------------------------
  # Internals
  # ------------------------------------------------------------------

  # Walk up to 5 redirects, then stream the final response straight to disk.
  # Uses an explicit X509 store with default paths because Net::HTTP's
  # built-in cert store fails CRL verification on some macOS / OpenSSL@3
  # setups (`certificate verify failed (unable to get certificate CRL)`).
  def download(url, dest, max_redirects: 5)
    uri  = URI(url)
    hops = 0
    while hops < max_redirects
      next_uri = nil
      http = Net::HTTP.new(uri.host, uri.port)
      if uri.scheme == "https"
        http.use_ssl    = true
        http.cert_store = OpenSSL::X509::Store.new.tap(&:set_default_paths)
      end
      http.start do
        req = Net::HTTP::Get.new(uri.request_uri, "User-Agent" => "toy-rnn/0.1")
        http.request(req) do |res|
          case res
          when Net::HTTPSuccess
            File.open(dest, "wb") { |f| res.read_body { |chunk| f.write(chunk) } }
            return dest
          when Net::HTTPRedirection
            next_uri = URI.join(uri.to_s, res["location"])
          else
            raise "fetch failed (#{res.code} #{res.message}): #{uri}"
          end
        end
      end
      raise "redirect with no Location: #{uri}" unless next_uri
      uri  = next_uri
      hops += 1
    end
    raise "too many redirects fetching #{url}"
  end
end
