// ESLint flat config (v9+).
//
// The `lint` script referenced eslint from the start, but no config file existed — so `make lint` and
// `make pr` failed at the dashboard step with a config-not-found error rather than a lint result. The
// Makefile's `[ -d node_modules ]` guard hid that for as long as deps were never installed.
//
// Type-aware rules are deliberately on: the dashboard consumes `NetworkState` over a WebSocket, and the
// mistakes worth catching there (a promise never awaited, a field narrowed from `unknown` incorrectly)
// are exactly the ones only type information can see.

import js from '@eslint/js';
import globals from 'globals';
import reactHooks from 'eslint-plugin-react-hooks';
import reactRefresh from 'eslint-plugin-react-refresh';
import tseslint from 'typescript-eslint';

export default tseslint.config(
  { ignores: ['dist', 'coverage', 'node_modules'] },
  {
    extends: [js.configs.recommended, ...tseslint.configs.recommendedTypeChecked],
    files: ['src/**/*.{ts,tsx}'],
    languageOptions: {
      ecmaVersion: 2022,
      globals: globals.browser,
      parserOptions: {
        project: ['./tsconfig.json'],
        tsconfigRootDir: import.meta.dirname,
      },
    },
    plugins: {
      'react-hooks': reactHooks,
      'react-refresh': reactRefresh,
    },
    rules: {
      ...reactHooks.configs.recommended.rules,
      'react-refresh/only-export-components': ['warn', { allowConstantExport: true }],

      // Mirrors the Python side's no-`Any` rule. The stream payload arrives as `unknown` and must be
      // narrowed, not asserted — an `any` here is how a renamed field reaches the DOM as `undefined`.
      '@typescript-eslint/no-explicit-any': 'error',
      '@typescript-eslint/no-unsafe-assignment': 'error',
      '@typescript-eslint/no-unsafe-member-access': 'error',

      // A dropped promise in a WebSocket handler fails silently and stalls the live view.
      '@typescript-eslint/no-floating-promises': 'error',

      '@typescript-eslint/no-unused-vars': [
        'error',
        { argsIgnorePattern: '^_', varsIgnorePattern: '^_' },
      ],
    },
  },
  {
    // Tests run in jsdom and legitimately reach for non-null assertions on rendered nodes.
    files: ['src/**/*.test.{ts,tsx}'],
    rules: {
      '@typescript-eslint/no-non-null-assertion': 'off',
    },
  },
);
