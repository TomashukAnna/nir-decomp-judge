// OpenCode 1.19.4 otherwise adds a 32,000-token cap to each model response.
export default async () => ({
  "chat.params": async (_input, output) => {
    delete output.maxOutputTokens;
  },
});
