---
id: what-are-chains
---

# What are LLM chains and why are they useful?

Chains are a concept in the world of language models designed to model common patterns for applying large language models (LLMs) to a sequence of tasks. Although the term "chain" might suggest that it strictly involves chaining together LLM steps, the name has stuck, and it is now used more broadly.

Chains provide a convenient abstraction for organizing and executing a series of LLM steps in various ways to achieve desired outcomes. `llm-chain` ships two chain types: **Sequential** and **Map-Reduce**. Both are generic over the driver, so the same chain code works against OpenAI, Anthropic, Gemini, Bedrock, Ollama, or a local llama.cpp model.

## Sequential Chains

Sequential chains are a simple yet powerful approach to applying LLMs. They connect multiple steps together in a sequence, where the output of the first step becomes the input of the second step, and so on. This method allows for straightforward processing of information, where each step builds upon the results of the previous one.

## Map-Reduce Chains

Map-reduce chains are designed to work with one or more documents. They apply the map prompt to each document in parallel, combine the results, and then run the reduce prompt over the combination to produce a final output.

This approach is particularly useful when working with large documents or multiple documents, as it enables parallel processing and efficient combination of results.

## Errors are typed

Running a chain returns `Result<Output, ChainError>`. `ChainError::Format` means a prompt template referenced a missing parameter, `ChainError::Execute` wraps the driver's typed API error, and `ChainError::Empty` signals an empty chain or document list. Nothing panics.

In summary, chains are a useful concept in applying LLMs, as they provide a structured way of organizing and executing LLM steps for various tasks. Each chain type has its unique characteristics and advantages, and choosing the right chain for your specific use case can significantly improve the effectiveness of your LLM application.
