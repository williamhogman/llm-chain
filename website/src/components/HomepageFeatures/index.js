import clsx from "clsx";
import React from "react";
import styles from "./styles.module.css";

const FeatureList = [
  {
    title: "Every major provider",
    description: (
      <>
        OpenAI, Anthropic, Gemini, AWS Bedrock, Ollama and local llama.cpp
        (GGUF) — all behind one typed, native-async API. Switching providers
        is a matter of swapping imports.
      </>
    ),
  },
  {
    title: "Chains",
    description: (
      <>
        Build powerful chains of prompts that allow you to execute more complex
        tasks, step by step, leveraging the full potential of LLMs.
      </>
    ),
  },
  {
    title: "Tools",
    description: (
      <>
        Unleash LLMs in the real world with a set of tools that allow your LLMs
        to perform actions like running Python code.
      </>
    ),
  },
];

function Feature({ title, description }) {
  return (
    <div className={clsx("col col--4")}>
      <div className="text--center padding-horiz--md">
        <h3>{title}</h3>
        <p>{description}</p>
      </div>
    </div>
  );
}

export default function HomepageFeatures() {
  return (
    <section className={styles.features}>
      <div className="container">
        <div className="row">
          {FeatureList.map((props, idx) => (
            <Feature key={idx} {...props} />
          ))}
        </div>
      </div>
    </section>
  );
}
