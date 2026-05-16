#!/usr/bin/env ruby
# frozen_string_literal: true
# tep_demo/chat.rb — a small CLI chat client for the OpenAI-shaped server.
#
# Talks to tep_demo/openai_api via POST /v1/chat/completions. Holds
# the conversation history across turns and sends the full thread
# each time (the API is stateless; the client owns the history).
#
# Run the server first:
#   ./prep/build_tep_app.sh tep_demo/openai_api.rb tep_demo/openai_api
#   ./tep_demo/openai_api -p 4567 &
#
# Then this client (CRuby, host-side — not Spinel-compiled):
#   ./tep_demo/chat.rb
#
# Env vars: TOY_API (default http://127.0.0.1:4567/v1),
#           TOY_MODEL (default gpt2),
#           TOY_MAX_TOKENS (default 60).
#
# Slash commands at the prompt:
#   /reset   clear history
#   /system  set the system prompt (stays at the top of every request)
#   /tokens  set max_tokens
#   /quit    exit

require "net/http"
require "json"
require "uri"

API_URL = ENV.fetch("TOY_API",        "http://127.0.0.1:4567/v1")
MODEL   = ENV.fetch("TOY_MODEL",      "gpt2")
MAX_TOK = (ENV["TOY_MAX_TOKENS"] || "60").to_i

class Chat
  attr_accessor :max_tokens, :system_prompt

  def initialize
    @history       = []                # list of {role:, content:}
    @system_prompt = nil
    @max_tokens    = MAX_TOK
  end

  def reset!
    @history = []
  end

  def turns
    @history.size / 2  # one user + one assistant per turn
  end

  def send_user(text)
    @history << { role: "user", content: text }
    messages = @system_prompt ?
      [{ role: "system", content: @system_prompt }] + @history :
      @history
    reply = post_chat(messages)
    @history << { role: "assistant", content: reply }
    reply
  end

  private

  def post_chat(messages)
    uri = URI.parse("#{API_URL}/chat/completions")
    req = Net::HTTP::Post.new(uri, "Content-Type" => "application/json")
    req.body = JSON.dump(
      model:      MODEL,
      messages:   messages,
      max_tokens: @max_tokens,
    )
    res = Net::HTTP.start(uri.hostname, uri.port,
                           read_timeout: 600) { |http| http.request(req) }
    body = JSON.parse(res.body)
    raise body.dig("error", "message") || res.body if res.code.to_i >= 400
    body.dig("choices", 0, "message", "content") or
      raise "no message.content in response: #{body.inspect}"
  end
end

def banner(chat)
  puts "─ toy/chat ──────────────────────────────────────────────────────"
  puts "  endpoint   #{API_URL}"
  puts "  model      #{MODEL}"
  puts "  max_tokens #{chat.max_tokens}  (/tokens N to change)"
  puts "  history    cleared  (/reset to clear, /system <text>, /quit to exit)"
  puts "─────────────────────────────────────────────────────────────────"
end

def repl
  chat = Chat.new
  banner(chat)
  loop do
    print "\n> "
    $stdout.flush
    line = $stdin.gets&.chomp
    break if line.nil? || line == "/quit"
    next  if line.strip.empty?

    case line
    when "/reset"
      chat.reset!
      puts "(history cleared, #{chat.turns} turns)"
      next
    when /\A\/system\s+(.+)\z/m
      chat.system_prompt = $1.strip
      puts "(system prompt set: #{chat.system_prompt.inspect})"
      next
    when /\A\/tokens\s+(\d+)\z/
      chat.max_tokens = Integer($1)
      puts "(max_tokens = #{chat.max_tokens})"
      next
    when %r{\A/}
      puts "(unknown command: #{line})"
      next
    end

    begin
      t0    = Time.now
      reply = chat.send_user(line)
      dt    = ((Time.now - t0) * 1000).round
      puts reply.strip
      puts "  [turn #{chat.turns}, #{dt} ms]"
    rescue StandardError, Interrupt => e
      puts "  (error: #{e.message})"
    end
  end
  puts "\nbye."
end

repl if $PROGRAM_NAME == __FILE__
