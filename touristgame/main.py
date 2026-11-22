from touristgame.montecarlo import MonteCarloSimulation

MonteCarloSimulation(
    size=100, scenario="Scenario 1", reputation_threshold=5.0, max_steps=10
).run()
