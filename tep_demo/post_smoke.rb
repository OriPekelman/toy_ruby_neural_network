# Minimum Tep POST smoke. Helps isolate whether the openai_api POST
# issue is in the project's handler or in the Tep POST path itself.
require 'sinatra'

get '/' do
  "hello\n"
end

post '/echo' do
  res.headers["Content-Type"] = "application/json"
  body = req.body
  "{\"got\":\"" + body + "\",\"len\":" + body.length.to_s + "}\n"
end

post '/v1/chat/completions' do
  res.headers["Content-Type"] = "application/json"
  "{\"object\":\"chat.completion\",\"choices\":[{\"message\":{\"role\":\"assistant\",\"content\":\"hello world\"}}]}\n"
end
