import json
from typing import Any

from tonal_hortator.core.database.queries import GET_USER_FEEDBACK_FOR_TRAINING


class TrainingDataAggregator:
    """
    Aggregates user feedback into structured training examples
    for LLM prompt-tuning or supervised fine-tuning.
    """

    RATING_THRESHOLD = 4

    def __init__(self, db: Any) -> None:
        self.db = db

    def aggregate(self, output_path: str = "playlist_training_data.jsonl") -> None:
        feedback_rows = self.db.execute_fetchall(GET_USER_FEEDBACK_FOR_TRAINING)
        examples = []
        for row in feedback_rows:
            (
                query,
                query_type,
                parsed_genres,
                parsed_mood,
                tracks_json,
                user_rating,
                user_actions_json,
            ) = row
            label = 1 if user_rating and user_rating >= self.RATING_THRESHOLD else 0
            example = {
                "input": query,
                "system_parsed": {
                    "query_type": query_type,
                    "genres": json.loads(parsed_genres) if parsed_genres else [],
                    "mood": parsed_mood,
                },
                "playlist": json.loads(tracks_json) if tracks_json else [],
                "user_feedback": {
                    "rating": user_rating,
                    "actions": (
                        json.loads(user_actions_json) if user_actions_json else []
                    ),
                },
                "label": label,
            }
            examples.append(example)
        with open(output_path, "w") as f:
            for ex in examples:
                f.write(json.dumps(ex) + "\n")
        print(f"Wrote {len(examples)} training examples to {output_path}")


# Optional CLI entry point for easy running
if __name__ == "__main__":
    from tonal_hortator.core.database import DatabaseManager

    db = DatabaseManager("music_library.db")
    aggregator = TrainingDataAggregator(db)
    aggregator.aggregate()
