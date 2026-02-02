"""Database module for storing red cell labels."""

import sqlite3
from pathlib import Path
from typing import Optional, List, Tuple
from datetime import datetime


class LabelDatabase:
    """SQLite database for storing red cell labels."""

    def __init__(self, db_path: Path):
        """Initialize the database.

        Parameters
        ----------
        db_path : Path
            Path to the SQLite database file.
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self):
        """Initialize the database schema."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Create labels table
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS labels (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                annotator_name TEXT NOT NULL,
                roi_id INTEGER NOT NULL,
                is_red INTEGER NOT NULL,
                timestamp TEXT NOT NULL,
                UNIQUE(annotator_name, roi_id)
            )
            """
        )

        # Create index for faster lookups
        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_annotator_roi 
            ON labels(annotator_name, roi_id)
            """
        )

        conn.commit()
        conn.close()

    def add_label(self, annotator_name: str, roi_id: int, is_red: bool) -> bool:
        """Add or update a label.

        Parameters
        ----------
        annotator_name : str
            Name of the annotator.
        roi_id : int
            ID of the ROI being labeled.
        is_red : bool
            True if the ROI is red, False otherwise.

        Returns
        -------
        bool
            True if label was added, False if it was updated.
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        timestamp = datetime.now().isoformat()

        # Try to insert, if it exists, update
        cursor.execute(
            """
            INSERT OR REPLACE INTO labels (annotator_name, roi_id, is_red, timestamp)
            VALUES (?, ?, ?, ?)
            """,
            (annotator_name, roi_id, int(is_red), timestamp),
        )

        conn.commit()
        conn.close()
        return True

    def get_label(self, annotator_name: str, roi_id: int) -> Optional[bool]:
        """Get a label for a specific ROI.

        Parameters
        ----------
        annotator_name : str
            Name of the annotator.
        roi_id : int
            ID of the ROI.

        Returns
        -------
        Optional[bool]
            True if red, False if not red, None if not labeled.
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT is_red FROM labels
            WHERE annotator_name = ? AND roi_id = ?
            """,
            (annotator_name, roi_id),
        )

        result = cursor.fetchone()
        conn.close()

        if result is None:
            return None
        return bool(result[0])

    def get_labeled_rois(self, annotator_name: str) -> List[int]:
        """Get list of ROI IDs that have been labeled by an annotator.

        Parameters
        ----------
        annotator_name : str
            Name of the annotator.

        Returns
        -------
        List[int]
            List of ROI IDs that have been labeled.
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT roi_id FROM labels
            WHERE annotator_name = ?
            """,
            (annotator_name,),
        )

        results = cursor.fetchall()
        conn.close()

        return [row[0] for row in results]

    def get_stats(self, annotator_name: Optional[str] = None) -> dict:
        """Get labeling statistics.

        Parameters
        ----------
        annotator_name : Optional[str]
            If provided, get stats for this annotator only.
            If None, get stats for all annotators.

        Returns
        -------
        dict
            Dictionary with statistics:
            - total_labels: total number of labels
            - red_count: number of red labels
            - not_red_count: number of not red labels
            - unique_rois: number of unique ROIs labeled
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        if annotator_name:
            cursor.execute(
                """
                SELECT COUNT(*), SUM(is_red), COUNT(DISTINCT roi_id)
                FROM labels
                WHERE annotator_name = ?
                """,
                (annotator_name,),
            )
        else:
            cursor.execute(
                """
                SELECT COUNT(*), SUM(is_red), COUNT(DISTINCT roi_id)
                FROM labels
                """
            )

        total, red_sum, unique_rois = cursor.fetchone()
        conn.close()

        red_count = red_sum if red_sum is not None else 0
        not_red_count = total - red_count if total else 0

        return {
            "total_labels": total or 0,
            "red_count": red_count,
            "not_red_count": not_red_count,
            "unique_rois": unique_rois or 0,
        }
