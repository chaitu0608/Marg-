// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

/// @notice Stores an optional, tamper-evident pointer to an off-chain Marg run.
/// @dev This registry must never participate in payment, fraud, or routing decisions.
contract MargResultRegistry {
    struct Receipt {
        bytes32 sourceDigest;
        address publisher;
        uint64 recordedAt;
    }

    mapping(bytes32 resultDigest => Receipt) public receipts;

    error EmptyDigest();
    error AlreadyRegistered();

    event ResultRegistered(
        bytes32 indexed resultDigest, bytes32 indexed sourceDigest, address indexed publisher, uint64 recordedAt
    );

    /// @notice Permanently records hashes; the underlying files remain off-chain.
    function register(bytes32 resultDigest, bytes32 sourceDigest) external {
        if (resultDigest == bytes32(0) || sourceDigest == bytes32(0)) {
            revert EmptyDigest();
        }
        if (receipts[resultDigest].publisher != address(0)) {
            revert AlreadyRegistered();
        }

        uint64 recordedAt = uint64(block.timestamp);
        receipts[resultDigest] = Receipt(sourceDigest, msg.sender, recordedAt);
        emit ResultRegistered(resultDigest, sourceDigest, msg.sender, recordedAt);
    }

    /// @notice Returns true when an identical result bundle was registered before.
    function exists(bytes32 resultDigest) external view returns (bool) {
        return receipts[resultDigest].publisher != address(0);
    }
}
