import socketserver
from functools import partial
from http.server import SimpleHTTPRequestHandler

import click

from envision.server import run


@click.group(name="envision")
def envision_cli():
    pass


@envision_cli.command(name="start")
@click.option("-p", "--port", default=8081)
@click.option("-s", "--scenarios", multiple=True, default=["scenarios"])
def start_server(port, scenarios):
    run(scenarios, port)


envision_cli.add_command(start_server)
