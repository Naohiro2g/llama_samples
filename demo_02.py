from llama_cpp import Llama

llm = Llama(
    model_path="models/Llama-3-ELYZA-JP-8B-q4_k_m.gguf",
    chat_format="llama-3",
    n_ctx=1024,
)

response = llm.create_chat_completion(
    messages=[
        {
            "role": "system",
            "content": "あなたは誠実で優秀な日本人のアシスタントです。特に指示が無い場合は、常に日本語で回答してください。",
        },
        {
            "role": "user",
            "content": "以下の文章を要約してください。\
            # 文章\
            エッジコンピューティング（英語: Edge computing）とは、利用者や端末と物理的に近い場所に処理装置（エッジプラットフォーム）を分散配置して、ネットワークの端点でデータ処理を行う技術の総称。多くのデバイスが接続されるIoT時代となり提唱されるようになった。\
            エッジコンピューティングは分散コンピューティングの活用であり、サーバ処理とデータストレージをリクエスト元にネットワーク上距離を近づける事で、処理応答時間を改善し、バックボーン帯域幅を節約する事に寄与する。\
            エッジコンピューティングの起源は、要求元ユーザーの近くに配置されたエッジサーバーからWebおよびストリームコンテンツを提供するために1990年代後半に開始されたコンテンツデリバリネットワーク(CDN)にある。近年これらのネットワークはエッジサーバーでアプリケーションとアプリケーションコンポーネントをホストするよう進化し、リアルタイムデータ処理などのアプリケーションをホストするIoTを見越した最新のエッジコンピューティングは、仮想化テクノロジーを通じてこのアプローチを大幅に拡張し、エッジサーバーでの幅広いアプリケーションの展開と実行を容易にしている。",
        },
    ],
    max_tokens=1024,
)

print(response["choices"][0]["message"]["content"])
