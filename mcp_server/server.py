from mcp.server.mcpserver import MCPServer

from mcp_server.tools.generate_image import register as register_generate_image

mcp = MCPServer("art-director-agent")
register_generate_image(mcp)


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
