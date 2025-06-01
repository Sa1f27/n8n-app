CREATE TABLE IF NOT EXISTS visited_sites (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    url TEXT NOT NULL UNIQUE,
    title TEXT,
    last_visit_time DATETIME,
    html_content TEXT
);