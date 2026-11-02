import pandas as pd

class ContextFilter:
    """
    Context-Aware Recommendation filter and score modifier.
    Adjusts item relevance based on temporal context (day, time) and hardware device.
    """
    def __init__(self, movies_df: pd.DataFrame):
        self.movies_df = movies_df.copy().set_index('movie_id')

    def compute_context_modifier(
        self,
        movie_id: int,
        device: str = "smart_tv",
        time_of_day: str = "evening",
        day_of_week: str = "friday",
        genre_filter: str = None
    ) -> float:
        """
        Calculates an additive contextual boost score [-0.2, +0.25] for a given movie.
        """
        if movie_id not in self.movies_df.index:
            return 0.0

        row = self.movies_df.loc[movie_id]
        genres = row['genres'].lower()
        runtime = row['runtime_min']
        boost = 0.0

        # 1. Device Context
        if device == "mobile":
            # Mobile users strongly favor shorter runtimes & light entertainment
            if runtime <= 115:
                boost += 0.10
            elif runtime >= 150:
                boost -= 0.15
            if any(g in genres for g in ['comedy', 'animation']):
                boost += 0.08
        elif device == "smart_tv":
            # Big screen favors immersive epics, high visual fidelity & thrillers
            if runtime >= 135:
                boost += 0.08
            if any(g in genres for g in ['sci-fi', 'action', 'adventure', 'crime']):
                boost += 0.10

        # 2. Temporal Context (Time of day & Day of week)
        is_weekend = day_of_week.lower() in ['friday', 'saturday', 'sunday']

        if time_of_day == "night" and is_weekend:
            # Prime-time blockbuster or intense drama
            if any(g in genres for g in ['thriller', 'sci-fi', 'mystery', 'crime']):
                boost += 0.12
        elif time_of_day in ["morning", "afternoon"] and not is_weekend:
            # Daytime weekday: preference for lighter, compact movies
            if runtime <= 110:
                boost += 0.12
            if 'drama' in genres and runtime > 150:
                boost -= 0.10

        # 3. Explicit Genre Focus (Knowledge-based constraint if requested)
        if genre_filter and genre_filter.lower() != "all":
            if genre_filter.lower() in genres:
                boost += 0.20
            else:
                boost -= 0.30

        return boost
