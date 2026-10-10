export default [
  {
    files: ["extension/**/*.js"],
    languageOptions: {
      ecmaVersion: 2022,
      sourceType: "module",
      globals: {
        chrome: "readonly",
        importScripts: "readonly",
        window: "readonly",
        document: "readonly",
        console: "readonly",
        setTimeout: "readonly",
        clearTimeout: "readonly",
        setInterval: "readonly",
        clearInterval: "readonly",
        fetch: "readonly",
        AbortController: "readonly",
        HTMLElement: "readonly",
        Range: "readonly",
        Node: "readonly",
        DOMParser: "readonly",
        module: "readonly",
        exports: "readonly"
      }
    },
    rules: {
      "no-undef": "error",
      "no-unused-vars": [
        "warn",
        {
          "argsIgnorePattern": "^_",
          "caughtErrors": "none",
          "ignoreRestSiblings": true
        }
      ]
    }
  }
];
