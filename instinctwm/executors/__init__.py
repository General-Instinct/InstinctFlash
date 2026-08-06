"""Executors — carry out a `Plan` against a live server.

This is the only layer permitted to touch the model: bind arguments, install rewrites, replay
captured graphs. Planners decide; executors act. Keeping the seam sharp is what makes a plan
inspectable before anything is loaded.
"""
