import cobra
from glocom import GloCom, Experiment


def medium(**uptake_limits):
    """Build a medium: each keyword is a shared metabolite id, each value the max uptake rate.
    Exchange reactions in the community model are named 'EX#<metabolite>'.
    Negative lower bound = uptake allowed (e.g. -10 means up to 10 units/h of import)."""
    return [
        cobra.Reaction(f"EX#{met}", lower_bound=-limit, upper_bound=1000.0)
        for met, limit in uptake_limits.items()
    ]


# Made-up experimental data (growth = True, no growth = False), for the target organism OrgA
experiments = [
    Experiment(reactions=medium(glc_e=10), growth={"OrgA": True}),   # exp 0: glucose  -> grows
    Experiment(reactions=medium(xyl_e=10), growth={"OrgA": True}),   # exp 1: xylose   -> grows
    Experiment(reactions=medium(ace_e=10), growth={"OrgA": False}),  # exp 2: acetate  -> no growth
]

# The answer GloCom should find (format: "ACTION reaction_id", comma separated).
# With the toy's default weights this costs 4 (add) + 3 (remove) = 7.
EXPECTED = "ADD OrgA#DB#XYL2GLC, REM OrgA#ACEG"

if __name__ == "__main__":
    glocom = GloCom.from_list(
        target_org="OrgA",
        org_list=["OrgA"],
        experiments=experiments,
        expected_result=EXPECTED,
    )
    problem, passed = glocom.run()

    print("\n================ TOY EXAMPLE SUMMARY ================")
    print(f"Solver status : {problem.status}")
    print(f"Objective     : {problem.value}")
    print(f"Expected fix  : {EXPECTED}")
    print(f"Result        : {'PASS - GloCom found the expected modifications' if passed else 'FAIL - see glocom.log'}")
    exit(0 if passed else 1)
