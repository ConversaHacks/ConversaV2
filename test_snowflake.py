#!/usr/bin/env python3
"""
Snowflake AI_TRANSCRIBE with speaker diarization.

Usage:
  python test_snowflake.py <local_file>       # Upload (if needed) and transcribe
  python test_snowflake.py <local_file> --raw # Show raw JSON output (for debugging)
  python test_snowflake.py --list             # List files in stage
  python test_snowflake.py --create-stage     # Create the stage if it doesn't exist

Can also be imported and used as a module:
  from test_snowflake import transcribe_audio
  result = transcribe_audio("path/to/audio.wav")
"""

import os
import sys
from typing import Optional

try:
    from snowflake.connector import connect
except ImportError:
    print("Error: snowflake-connector-python not installed")
    print("Run: pip install snowflake-connector-python")
    exit(1)

# Snowflake connection settings (config vars)
SNOWFLAKE_ACCOUNT = os.getenv("SNOWFLAKE_ACCOUNT", "TGFRRXQ-CSB74163")
SNOWFLAKE_USER = os.getenv("SNOWFLAKE_USER", "GONSIS10")
SNOWFLAKE_PASSWORD = os.getenv("SNOWFLAKE_PASSWORD", "Overrated1010@")
SNOWFLAKE_ROLE = os.getenv("SNOWFLAKE_ROLE", "ACCOUNTADMIN")
SNOWFLAKE_WAREHOUSE = os.getenv("SNOWFLAKE_WAREHOUSE", "COMPUTE_WH")
SNOWFLAKE_DATABASE = os.getenv("SNOWFLAKE_DATABASE", "AUDIO_DB")
SNOWFLAKE_SCHEMA = os.getenv("SNOWFLAKE_SCHEMA", "PUBLIC")
STAGE = os.getenv("SNOWFLAKE_STAGE", "@audio_stage")

# Build config dict based on auth method
SNOWFLAKE_CONFIG = {
    "account": SNOWFLAKE_ACCOUNT,
    "user": SNOWFLAKE_USER,
    "role": SNOWFLAKE_ROLE,
    "warehouse": SNOWFLAKE_WAREHOUSE,
    "database": SNOWFLAKE_DATABASE,
    "schema": SNOWFLAKE_SCHEMA,
    "client_session_keep_alive": True,
}

# Use password auth if password is set, otherwise externalbrowser
if SNOWFLAKE_PASSWORD:
    SNOWFLAKE_CONFIG["password"] = SNOWFLAKE_PASSWORD
    AUTH_METHOD = "password"
else:
    SNOWFLAKE_CONFIG["authenticator"] = "externalbrowser"
    AUTH_METHOD = "externalbrowser (SSO)"


def get_connection():
    """Connect to Snowflake."""
    return connect(**SNOWFLAKE_CONFIG)


def get_staged_files(connection) -> set:
    """Get set of filenames in the stage."""
    cur = connection.cursor()
    cur.execute(f"LIST {STAGE}")
    rows = cur.fetchall()
    # Extract just the filename from the full path (e.g., "audio_stage/file.wav" -> "file.wav")
    return {os.path.basename(row[0]) for row in rows}


def create_stage(connection, force: bool = False):
    """Create the stage without encryption (required for AI_TRANSCRIBE)."""
    stage_name = STAGE.lstrip("@")
    cur = connection.cursor()

    if force:
        print(f"[Snowflake] Dropping existing stage {stage_name}...")
        cur.execute(f"DROP STAGE IF EXISTS {stage_name}")

    print(f"[Snowflake] Creating stage {stage_name} (no encryption)...")
    # ENCRYPTION = (TYPE = 'SNOWFLAKE_SSE') disables client-side encryption
    cur.execute(f"CREATE STAGE IF NOT EXISTS {stage_name} ENCRYPTION = (TYPE = 'SNOWFLAKE_SSE')")
    print(f"[Snowflake] Stage {stage_name} ready")


def list_files(connection):
    """List files in the stage."""
    cur = connection.cursor()
    try:
        cur.execute(f"LIST {STAGE}")
        rows = cur.fetchall()
    except Exception as e:
        if "does not exist" in str(e):
            print(f"\nStage {STAGE} does not exist.")
            print("Run: python test_snowflake.py --create-stage")
            return
        raise

    print(f"\nFiles in {STAGE}:")
    print("=" * 60)
    if not rows:
        print("(empty)")
    else:
        for row in rows:
            print(f"  {row[0]} ({row[1]} bytes)")
    print()


def upload_file(connection, local_path: str) -> bool:
    """Upload a local file to the stage. Returns True on success."""
    if not os.path.exists(local_path):
        print(f"[Snowflake] Error: File not found: {local_path}")
        return False

    abs_path = os.path.abspath(local_path)
    print(f"[Snowflake] Uploading {os.path.basename(local_path)} to {STAGE}...")

    cur = connection.cursor()
    cur.execute(f"PUT 'file://{abs_path}' {STAGE} AUTO_COMPRESS=FALSE OVERWRITE=TRUE")
    rows = cur.fetchall()

    for row in rows:
        status = row[5] if len(row) > 5 else "OK"
        print(f"[Snowflake] Upload: {row[0]} -> {status}")
    return True


def transcribe_file_raw(connection, file_name: str) -> dict:
    """Get raw AI_TRANSCRIBE output for debugging."""
    sql = f"""
    SELECT AI_TRANSCRIBE(
      TO_FILE('{STAGE}', '{file_name}'),
      {{'timestamp_granularity': 'speaker'}}
    ) AS result;
    """
    cur = connection.cursor()
    cur.execute(sql)
    row = cur.fetchone()
    return row[0] if row else None


def transcribe_file(connection, file_name: str) -> list:
    """
    Transcribe a file from the stage.
    Returns list of (speaker, start_s, end_s, text) tuples.
    """
    sql = f"""
    WITH t AS (
      SELECT AI_TRANSCRIBE(
        TO_FILE('{STAGE}', '{file_name}'),
        {{'timestamp_granularity': 'speaker'}}
      ) AS j
    )
    SELECT
      seg.value:speaker_label::string AS speaker,
      seg.value:start::float         AS start_s,
      seg.value:end::float           AS end_s,
      seg.value:text::string         AS text
    FROM t,
    LATERAL FLATTEN(input => j:segments) seg
    ORDER BY start_s;
    """

    cur = connection.cursor()
    cur.execute(sql)
    return cur.fetchall()


def transcribe_audio(local_path: str, verbose: bool = True) -> Optional[list]:
    """
    Main function: Upload (if needed) and transcribe an audio file.

    Args:
        local_path: Path to local audio file
        verbose: Print progress messages

    Returns:
        List of (speaker, start_s, end_s, text) tuples, or None on error
    """
    if not os.path.exists(local_path):
        if verbose:
            print(f"[Snowflake] Error: File not found: {local_path}")
        return None

    file_name = os.path.basename(local_path)

    try:
        if verbose:
            print(f"[Snowflake] Connecting ({AUTH_METHOD})...")
        connection = get_connection()
    except Exception as e:
        if verbose:
            print(f"[Snowflake] Connection failed: {e}")
        return None

    try:
        # Check if file exists in stage
        staged_files = get_staged_files(connection)

        if file_name not in staged_files:
            if verbose:
                print(f"[Snowflake] File not in stage, uploading...")
            if not upload_file(connection, local_path):
                return None
        else:
            if verbose:
                print(f"[Snowflake] File already in stage")

        # Transcribe
        if verbose:
            print(f"[Snowflake] Transcribing {file_name}...")

        result = transcribe_file(connection, file_name)

        if verbose:
            print(f"[Snowflake] Got {len(result)} segments")

        return result

    except Exception as e:
        if verbose:
            print(f"[Snowflake] Error: {e}")
        return None

    finally:
        connection.close()


def print_transcript(segments: list):
    """Print transcript in readable format."""
    print()
    print("=" * 60)
    print("TRANSCRIPT (with speaker diarization):")
    print("=" * 60)

    if not segments:
        print("No segments returned (no speech detected?)")
    else:
        for speaker, start_s, end_s, text in segments:
            print(f"[{start_s:8.2f} - {end_s:8.2f}] {speaker}: {text}")
        print()
        print(f"Total segments: {len(segments)}")


def main():
    args = sys.argv[1:]

    if "--help" in args or "-h" in args:
        print(__doc__)
        return

    raw_mode = "--raw" in args

    if "--create-stage" in args:
        print(f"Account: {SNOWFLAKE_ACCOUNT}")
        print(f"Database: {SNOWFLAKE_DATABASE}")
        force = "--force" in args
        try:
            connection = get_connection()
            create_stage(connection, force=force)
            connection.close()
        except Exception as e:
            print(f"Error: {e}")
        return

    if "--list" in args:
        print(f"Account: {SNOWFLAKE_ACCOUNT}")
        print(f"Stage: {STAGE}")
        try:
            connection = get_connection()
            list_files(connection)
            connection.close()
        except Exception as e:
            print(f"Error: {e}")
        return

    # Get file path from args
    local_path = None
    for arg in args:
        if not arg.startswith("--"):
            local_path = arg
            break

    if not local_path:
        print("Usage: python test_snowflake.py <audio_file>")
        print("       python test_snowflake.py --list")
        return

    # Transcribe
    if raw_mode:
        # Raw mode: show full JSON output for debugging
        import json
        file_name = os.path.basename(local_path)
        print(f"[Snowflake] Connecting ({AUTH_METHOD})...")
        try:
            connection = get_connection()
            staged_files = get_staged_files(connection)
            if file_name not in staged_files:
                print(f"[Snowflake] File not in stage, uploading...")
                upload_file(connection, local_path)
            print(f"[Snowflake] Getting raw AI_TRANSCRIBE output...")
            raw = transcribe_file_raw(connection, file_name)
            connection.close()
            print()
            print("=" * 60)
            print("RAW AI_TRANSCRIBE OUTPUT:")
            print("=" * 60)
            print(json.dumps(raw, indent=2))
        except Exception as e:
            print(f"Error: {e}")
    else:
        segments = transcribe_audio(local_path)
        if segments is not None:
            print_transcript(segments)


if __name__ == "__main__":
    main()