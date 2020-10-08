import argparse

from smarts.core.sumo_road_network import SumoRoadNetwork


def generate_glb_from_sumo_network(sumo_net_file, out_glb_file):
    road_network = SumoRoadNetwork.from_file(net_file=sumo_net_file)
    glb = road_network.build_glb(scale=1000)
    glb.write_glb(out_glb_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "sumo2mesh.py",
        description="Utility to export sumo road networks to mesh files.",
    )
    parser.add_argument("net", help="sumo net file (*.net.xml)", type=str)
    parser.add_argument("output_path", help="where to write the mesh file", type=str)
    args = parser.parse_args()

    generate_glb_from_sumo_network(args.net, args.output_path)
