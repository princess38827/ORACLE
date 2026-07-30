from evolving_robot.brain.planner import Planner, Sequence, Action, Condition, Status


planner = Planner()


def see_target(bb):
    return bb.get("visible", False)


def approach(bb):
    print("→ approaching")
    return Status.SUCCESS


def grasp(bb):
    print("→ grasping")
    return Status.SUCCESS


planner.register_goal(
    "pick",
    lambda bb: Sequence(
        [
            Condition(see_target, name="SeeTarget"),
            Action(approach, name="Approach"),
            Action(grasp, name="Grasp"),
        ],
        name="Pick",
    ),
)

planner.set_goal("pick")
print(planner.tick({"visible": False}).name)  # FAILURE
print(planner.tick({"visible": True}).name)  # SUCCESS (and runs the actions)
