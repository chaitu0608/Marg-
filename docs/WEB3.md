# Optional Web3 experiment receipt

## The useful Web3 boundary

Marg is a poor fit for putting payment decisions or fraud data on-chain. Payment authorization has strict latency, privacy, and operational requirements; an EVM transaction cannot safely sit in that request path. The useful, narrow experiment is a public, tamper-evident receipt for one reproducible artifact bundle.

The receipt contains only two SHA-256 digests: the saved benchmark JSON bundle and the relevant source snapshot. `MargResultRegistry` records the first publisher and block timestamp for each result digest, then emits an indexed event. It has no owner, update, delete, routing, scoring, or payment functions. The JSON, dataset, model, individual transactions, SHAP contributions, and keys stay off-chain.

```text
make verify
    ├── seeded simulation + model metrics + result JSON
    └── committed source and benchmark files
              │
              ▼
     scripts/create_attestation.py
     ├── SHA-256 of each results/*.json
     ├── result bundle SHA-256
     └── source snapshot SHA-256
              │ optional manual transaction
              ▼
     MargResultRegistry on local EVM / Sepolia / Ethereum mainnet
              │
              ▼
     indexed event + receipt lookup; no path back to scoring or routing
```

The hash commits to bytes; it does not prove that the simulator is realistic, that the code has no bugs, or that the reported result is suitable for production. Verification still requires obtaining the matching files and recomputing the hashes.

## Local preparation

Install Foundry using its official instructions, then compile and run the Solidity tests:

```sh
forge build
forge test -vv
make attest
cat artifacts/marg-attestation.json
```

`make attest` writes an ignored local manifest. Commit code and regenerate benchmark results before making the manifest; do not put private or customer data in the result files. The contract accepts the two `0x`-prefixed 32-byte digest values printed in the manifest.

To exercise deployment locally, start Anvil in one terminal and deploy from another with an Anvil-only key:

```sh
anvil
forge create contracts/src/MargResultRegistry.sol:MargResultRegistry \
  --rpc-url http://127.0.0.1:8545 \
  --private-key <ANVIL_TEST_KEY>
```

Use a Foundry test account or a local-only Anvil key. Never reuse that key outside the local chain.

## Public testnet and mainnet

For Sepolia or mainnet, compile and test locally first, inspect the bytecode and constructor (there are no constructor parameters), verify the RPC chain ID, and use a hardware-backed or Foundry-keystore account. The submitter then runs:

```sh
forge create contracts/src/MargResultRegistry.sol:MargResultRegistry \
  --rpc-url "$EVM_RPC_URL" \
  --account marg-attestor \
  --sender "$EVM_DEPLOYER_ADDRESS" \
  --broadcast
```

After deployment, publish the result receipt from the same account:

```sh
cast send "$MARG_REGISTRY_ADDRESS" \
  "register(bytes32,bytes32)" \
  "$MARG_RESULT_DIGEST" "$MARG_SOURCE_DIGEST" \
  --rpc-url "$EVM_RPC_URL" \
  --account marg-attestor \
  --sender "$EVM_DEPLOYER_ADDRESS" \
  --broadcast
```

Check the network and estimated gas before signing, then verify `exists(resultDigest)` and read `receipts(resultDigest)` after confirmation. A public chain transaction costs real gas and cannot be undone. No registry address or live deployment is claimed by this repository. Mainnet deployment requires the owner to choose the network, fund and control a deployer, review the bytecode, and personally authorize signing/broadcasting; this project has not done those steps.

## Limits and future value

- A public digest is permanent metadata and can reveal when a result bundle was published. It cannot be deleted.
- SHA-256 hashes establish byte equality only; they are not zero-knowledge proofs and do not hide the digest.
- This is an educational audit receipt, not a payment rail, settlement mechanism, token, or fraud oracle.
- A meaningful extension would be signed CI attestations with an independently verifiable build provenance statement, still without allowing chain state to influence transaction decisions.
