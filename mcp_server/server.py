from mcp.server.mcpserver import MCPServer

from mcp_server.tools.generate_image import register as register_generate_image
from mcp_server.tools.get_history import register as register_get_history
from mcp_server.tools.inpaint import register as register_inpaint
from mcp_server.tools.score_image import register as register_score_image
from mcp_server.tools.upscale import register as register_upscale

mcp = MCPServer("art-director-agent")
register_generate_image(mcp)
register_inpaint(mcp)
register_upscale(mcp)
register_score_image(mcp)
register_get_history(mcp)


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
