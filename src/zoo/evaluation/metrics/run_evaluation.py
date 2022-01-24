import argparse

from zoo.evaluation.metrics.collision.collision_data_parse import CollisionEvaluation
from zoo.evaluation.metrics.diversity.diversity_evaluation import DiversityEvaluation
from zoo.evaluation.metrics.kinematics.kinematics_evaluation import KinematicsEvaluation
from zoo.evaluation.metrics.offroad.offroad_data_parse import OffroadEvaluation


def run_all_evaluation(
    scenario_path,
    json_file_path,
    diversity_path,
    agent_name_list,
    evaluation_type_list,
    agent_group,
):
    evaluation_items = {
        "diversity": DiversityEvaluation(
            scenario_path, diversity_path, agent_group, agent_name_list
        ),
        "offroad": OffroadEvaluation(scenario_path, json_file_path, agent_name_list),
        "collision": CollisionEvaluation(
            scenario_path, json_file_path, agent_name_list
        ),
        "kinematics": KinematicsEvaluation(
            scenario_path, json_file_path, agent_name_list
        ),
    }
    for evaluation_type in evaluation_type_list:
        evaluation_type_object = evaluation_items[evaluation_type]
        evaluation_type_object.run_evaluation()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="evaluation",
        description="Start run evaluation.",
    )
    parser.add_argument(
        "scenario_logs_path",
        help="The path to the run all evaluation origin data",
        type=str,
    )
    parser.add_argument(
        "result_path", help="The path to the run all evaluation results", type=str
    )
    parser.add_argument(
        "diversity_result_path",
        help="The path to the run diversity evaluation result",
        type=str,
    )
    parser.add_argument(
        "evaluation_type_list",
        help="Evaluation type include kinematics offroad collision diversity",
        type=str,
    )
    parser.add_argument("all_agent_name_list", help="All agent name list", type=str)
    parser.add_argument("agent_groups", help="Agent groups", type=str)
    args = parser.parse_args()
    run_all_evaluation(
        args.scenario_logs_path,
        args.result_path,
        args.diversity_result_path,
        eval(args.all_agent_name_list),
        eval(args.evaluation_type_list),
        eval(args.agent_groups),
    )
