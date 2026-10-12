// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

import {MargResultRegistry} from "../src/MargResultRegistry.sol";

interface Vm {
    function expectRevert(bytes4 selector) external;
}

contract MargResultRegistryTest {
    Vm private constant vm = Vm(address(uint160(uint256(keccak256("hevm cheat code")))));
    MargResultRegistry private registry;

    function setUp() public {
        registry = new MargResultRegistry();
    }

    function testRegistersAndVerifiesReceipt() public {
        bytes32 results = keccak256("saved-results");
        bytes32 source = keccak256("source-tree");

        registry.register(results, source);

        (bytes32 savedSource, address publisher, uint64 recordedAt) = registry.receipts(results);
        require(registry.exists(results), "receipt should exist");
        require(savedSource == source, "source digest should be retained");
        require(publisher == address(this), "publisher should be retained");
        require(recordedAt > 0, "record time should be retained");
    }

    function testRejectsEmptyDigest() public {
        vm.expectRevert(MargResultRegistry.EmptyDigest.selector);
        registry.register(bytes32(0), keccak256("source-tree"));
    }

    function testRejectsDuplicateResultDigest() public {
        bytes32 results = keccak256("saved-results");
        registry.register(results, keccak256("source-tree"));

        vm.expectRevert(MargResultRegistry.AlreadyRegistered.selector);
        registry.register(results, keccak256("changed-source"));
    }
}
