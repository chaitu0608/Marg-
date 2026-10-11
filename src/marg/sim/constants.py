GATEWAYS = ("gw_a", "gw_b", "gw_c", "gw_d")
BANKS = tuple(f"bank_{i}" for i in range(6))
METHODS = ("upi", "card", "netbanking")
CONTEXTS = tuple((bank, method) for bank in BANKS for method in METHODS)
