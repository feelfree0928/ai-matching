"""
Debug script: check what post_types and post_statuses exist in your WordPress DB.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dotenv import load_dotenv
import pymysql

load_dotenv()

conn = pymysql.connect(
    host=os.getenv("DB_HOST", "localhost"),
    port=int(os.getenv("DB_PORT", "3306")),
    user=os.getenv("DB_USER", ""),
    password=os.getenv("DB_PASS", ""),
    database=os.getenv("DB_NAME", ""),
    charset="utf8mb4",
    cursorclass=pymysql.cursors.DictCursor,
)

try:
    with conn.cursor() as cur:
        # Check all unique post_types
        cur.execute("SELECT DISTINCT post_type, COUNT(*) as count FROM wp_posts GROUP BY post_type ORDER BY count DESC")
        print("=== Post Types in Database ===")
        for row in cur.fetchall():
            print(f"  {row['post_type']}: {row['count']} posts")
        
        print("\n=== Post Statuses ===")
        cur.execute("SELECT DISTINCT post_status, COUNT(*) as count FROM wp_posts GROUP BY post_status ORDER BY count DESC")
        for row in cur.fetchall():
            print(f"  {row['post_status']}: {row['count']} posts")
        
        # Check for resume-like post_types
        print("\n=== Checking for 'resume' post_type ===")
        cur.execute("SELECT COUNT(*) as count FROM wp_posts WHERE post_type = 'resume'")
        resume_count = cur.fetchone()['count']
        print(f"  post_type='resume': {resume_count} posts")
        
        if resume_count == 0:
            print("\n=== Trying common alternatives ===")
            for alt_type in ['resume', 'noo_resume', 'job_resume', 'candidate', 'profile']:
                cur.execute("SELECT COUNT(*) as count FROM wp_posts WHERE post_type = %s", (alt_type,))
                count = cur.fetchone()['count']
                if count > 0:
                    print(f"  post_type='{alt_type}': {count} posts")
                    # Show sample post_statuses for this type
                    cur.execute("SELECT DISTINCT post_status FROM wp_posts WHERE post_type = %s LIMIT 10", (alt_type,))
                    statuses = [r['post_status'] for r in cur.fetchall()]
                    print(f"    Statuses: {', '.join(statuses)}")
        
        # Check if there are ANY posts with resume-related meta
        print("\n=== Posts with resume-related meta keys ===")
        cur.execute("""
            SELECT DISTINCT p.post_type, p.post_status, COUNT(*) as count
            FROM wp_posts p
            INNER JOIN wp_postmeta pm ON p.ID = pm.post_id
            WHERE pm.meta_key = '_noo_resume_field__taetigkeiten'
            GROUP BY p.post_type, p.post_status
            ORDER BY count DESC
        """)
        for row in cur.fetchall():
            print(f"  post_type='{row['post_type']}', post_status='{row['post_status']}': {row['count']} posts")
            
finally:
    conn.close()
