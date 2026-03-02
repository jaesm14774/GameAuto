"""Game registry — maps game names to their strategy and agent classes."""

GAME_REGISTRY = {
    "elftw": {
        "strategy_cls": "games.elftw.strategy.ElfTWStrategy",
        "agent_cls": "games.elftw.agent.ElfTWAgent",
    },
}
