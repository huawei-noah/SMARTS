
from examples import default_argument_parser, policies
from smarts.proto import learner_pb2, learner_pb2_grpc, observation_pb2
from smarts.core import observation as obs_util
from smarts.core import action as act_util
from smarts.core.utils.networking import find_free_port
from smarts.rpc import server as learner_agent


def learner_loop(addr, agent_specs):

    #Spawn a grpc server to host the learner
    conn = learner_agent.spawn(learner_agent.serve, addr[1])
    conn.stub.build(
        agent_pb2.Specification(
            payload=cloudpickle.dumps(
                agent_specs
            )
        )
    )


def main(config:dict):
    assert len(config['agent_policies']) == len(config['agent_ids']), "Length of `agent_policies` does not match that of `agent_ids`."

    agent_specs = {
        agent_id: getattr(policies, agent_policy).agent.create_agent_spec()
        for agent_id, agent_policy in zip(config['agent_ids'], config['agent_policies'])
    }

    learner_loop(
        (config['learner_address'],config['learner_port']),
        agent_specs
    )


if __name__ == "__main__":
    parser = default_argument_parser("distributed-learner")
    parser.add_argument(
        "--agent_policies",
        help="Learner agent policies to use. In multi-agent simulation, agents may have different policies.",
        type=str,
        nargs="+"
        default=['keep_lane', 'keep_lane'],
    )    
    parser.add_argument(
        "--agent_ids",
        help="List of string, representing agent names. Length of `agent_ids` should match that of `agent_policies`.",
        type=str,
        nargs="+",
        default=["Agent_007", "Agent_008"],
    )
    parser.add_argument(
        "--learner_address",
        help="Server IP address in which the learner (i.e., RL ego agent) runs.",
        type=str,
        default='localhost',
    )
    parser.add_argument(
        "--learner_port",
        help="Server port at which the learner (i.e., RL ego agent) runs.",
        type=int,
        default=6001,
        required=True,
    )
    parser.add_argument(
        "--max_episode_steps",
        help="Maximum number of steps per episode.",
        type=int,
        default=None,
    )
    config = parser.parse_args()

    main(vars(config))
