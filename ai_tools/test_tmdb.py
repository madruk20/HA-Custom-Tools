import urllib.request
import urllib.parse
import urllib.error
import json

# ==========================================
# 1. PASTE YOUR TMDB READ ACCESS TOKEN HERE
# ==========================================
TMDB_TOKEN = "eyJhbGciOiJIUzI1NiJ9.eyJhdWQiOiJlMTQ3ZjhhYzc0Yjk0ZDBmZTM1OWRjMjNlODdkZjE2YiIsIm5iZiI6MTc3OTQ0MjkzMy43NDEsInN1YiI6IjZhMTAyNGY1MzQ3ZjAwYTZmYTQwMmUyMiIsInNjb3BlcyI6WyJhcGlfcmVhZCJdLCJ2ZXJzaW9uIjoxfQ.gJ4u7Et3iOtwkIcpvEvnAjmm6xR8yetGRAb8493l8X0"

# The movie we are searching for
QUERY = "Interstellar"

def test_tmdb():
    print(f"🔍 Initiating TMDB API Test for: '{QUERY}'...")
    
    if TMDB_TOKEN == "PASTE_YOUR_LONG_TOKEN_HERE" or not TMDB_TOKEN:
        print("❌ ERROR: You forgot to paste your TMDB_TOKEN into the script!")
        return

    # URL-encode the query to handle spaces safely
    safe_query = urllib.parse.quote(QUERY)
    url = f"https://api.themoviedb.org/3/search/movie?query={safe_query}&include_adult=false&page=1"
    
    # Build the request with the required headers
    req = urllib.request.Request(url)
    req.add_header("Authorization", f"Bearer {TMDB_TOKEN}")
    req.add_header("Accept", "application/json")

    try:
        # Fire the request
        with urllib.request.urlopen(req) as response:
            status = response.getcode()
            data = json.loads(response.read().decode('utf-8'))
            
            print(f"✅ CONNECTION SUCCESS! HTTP Status: {status}")
            
            if data.get("results"):
                first_match = data["results"][0]
                print(f"🎬 Found Movie: '{first_match.get('title')}'")
                print(f"🆔 TMDB ID: {first_match.get('id')}")
                print(f"📅 Release Date: {first_match.get('release_date')}")
            else:
                print("⚠️ Connected to TMDB successfully, but no movies were found for that query.")

    except urllib.error.HTTPError as e:
        print(f"\n❌ HTTP ERROR: {e.code} - {e.reason}")
        if e.code == 401:
            print("   -> 401 UNAUTHORIZED: TMDB rejected your token.")
            print("   -> Make sure you copied the 'API Read Access Token' (v4 Bearer), NOT the 'API Key' (v3).")
            print("   -> Ensure there are no spaces before or after the token string.")
    except urllib.error.URLError as e:
        print(f"\n❌ NETWORK ERROR: {e.reason}")
        print("   -> Could not reach api.themoviedb.org. Check your DNS or firewall.")
    except Exception as e:
        print(f"\n❌ UNEXPECTED ERROR: {str(e)}")

if __name__ == "__main__":
    test_tmdb()