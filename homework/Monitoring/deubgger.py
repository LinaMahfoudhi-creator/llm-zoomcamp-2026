from gitsource import GithubRepositoryDataReader

COMMIT = "8c1834d"

reader = GithubRepositoryDataReader(
    repo_owner="DataTalksClub",
    repo_name="llm-zoomcamp",
    commit_id=COMMIT,
    allowed_extensions={"md"},
    filename_filter=lambda path: "/lessons/" in path,
)

files = list(reader.read())
print("NUM FILES:", len(files))
if files:
    doc = files[0].parse()
    print(doc.keys())
    print(doc.get("filename"))
    print(doc.get("content", "")[:200])