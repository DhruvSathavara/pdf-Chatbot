import { NextRequest } from "next/server";
import { PineconeClient } from "@pinecone-database/pinecone";
import { PineconeStore } from "langchain/vectorstores/pinecone";
import { OpenAIEmbeddings } from "langchain/embeddings/openai";
import { OpenAI } from "langchain/llms/openai";
import { VectorDBQAChain } from "langchain/chains";
import { StreamingTextResponse, LangChainStream } from "ai";
import { CallbackManager } from "langchain/callbacks";

export async function POST(request: NextRequest) {
  const body = await request.json();

  const { stream, handlers } = LangChainStream();

  const pineconeClient = new PineconeClient();
  await pineconeClient.init({
    apiKey: process.env.NEXT_PUBLIC_PINECONE_API_KEY ?? "",
    environment: "us-west1-gcp-free",
  });

  const pineconeIndex = pineconeClient.Index(
    process.env.NEXT_PUBLIC_PINECONE_INDEX_NAME as string
  );

  const vectorStore = await PineconeStore.fromExistingIndex(
    new OpenAIEmbeddings(),
    { pineconeIndex }
  );
  const model = new OpenAI({
    modelName: "gpt-3.5-turbo",
    streaming: true,
    callbackManager: CallbackManager.fromHandlers(handlers),
  });
  const chain = VectorDBQAChain.fromLLM(model, vectorStore, {
    k: 1,
    returnSourceDocuments: true,
  });

  chain.call({ query: body.prompt }).catch(console.error);

  return new StreamingTextResponse(stream);
}
