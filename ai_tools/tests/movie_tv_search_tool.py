
# TMDB_API_KEY=eyJhbGciOiJIUzI1NiJ9.eyJhdWQiOiJlMTQ3ZjhhYzc0Yjk0ZDBmZTM1OWRjMjNlODdkZjE2YiIsIm5iZiI6MTc3OTQ0MjkzMy43NDEsInN1YiI6IjZhMTAyNGY1MzQ3ZjAwYTZmYTQwMmUyMiIsInNjb3BlcyI6WyJhcGlfcmVhZCJdLCJ2ZXJzaW9uIjoxfQ.gJ4u7Et3iOtwkIcpvEvnAjmm6xR8yetGRAb8493l8X0

        # ------------------------------------------
        # PATH 1: DETAILED TMDB MEDIA LOOKUP
        # ------------------------------------------
        if category == "media" and tmdb_key:
            import urllib.parse
            import asyncio
            safe_query = urllib.parse.quote(query)
            
            tmdb_headers = {
                "Authorization": f"Bearer {tmdb_key}",
                "Accept": "application/json"
            }
            
            async with aiohttp.ClientSession() as session:
                try:
                    # Stage 1: Search BOTH endpoints simultaneously
                    movie_url = f"https://api.themoviedb.org/3/search/movie?query={safe_query}&include_adult=false&page=1"
                    tv_url = f"https://api.themoviedb.org/3/search/tv?query={safe_query}&include_adult=false&page=1"
                    
                    movie_task = session.get(movie_url, headers=tmdb_headers, timeout=4.0)
                    tv_task = session.get(tv_url, headers=tmdb_headers, timeout=4.0)
                    
                    movie_resp, tv_resp = await asyncio.gather(movie_task, tv_task)
                    
                    if movie_resp.status == 401 or tv_resp.status == 401:
                        return {"error": "TMDB Authentication failed. Your TMDB_API_KEY is invalid."}
                        
                    movie_data = await movie_resp.json() if movie_resp.status == 200 else {}
                    tv_data = await tv_resp.json() if tv_resp.status == 200 else {}
                    
                    best_match = None
                    media_type = None
                    
                    movie_results = movie_data.get("results", [])
                    tv_results = tv_data.get("results", [])
                    
                    # Logic: Prioritize exact title matches over partial matches
                    target_query = query.lower()
                    
                    for m in movie_results:
                        if m.get("title", "").lower() == target_query:
                            best_match = m
                            media_type = "movie"
                            break
                            
                    if not best_match:
                        for t in tv_results:
                            if t.get("name", "").lower() == target_query:
                                best_match = t
                                media_type = "tv"
                                break
                                
                    # If no exact match, fallback to the very first result from either
                    if not best_match:
                        if tv_results:
                            best_match = tv_results[0]
                            media_type = "tv"
                        elif movie_results:
                            best_match = movie_results[0]
                            media_type = "movie"

                    if not best_match:
                        return {"result": f"Could not locate a movie or TV show matching '{query}' on TMDB."}

                    media_id = best_match.get("id")

                    # Stage 2: Pull down deep structural details
                    details_url = f"https://api.themoviedb.org/3/{media_type}/{media_id}?language=en-US&append_to_response=watch/providers,credits"
                    
                    async with session.get(details_url, headers=tmdb_headers, timeout=4.0) as resp:
                        if resp.status == 200:
                            details = await resp.json()
                            
                            title = details.get("title") or details.get("name")
                            release_date = details.get("release_date") or details.get("first_air_date")
                            runtime = f"{details.get('runtime')} mins" if details.get("runtime") else f"{details.get('number_of_seasons')} Seasons"
                            genres = [g.get("name") for g in details.get("genres", [])]
                            
                            # Extract Streaming Providers
                            watch_data = details.get("watch/providers", {}).get("results", {}).get("US", {})
                            stream_list = watch_data.get("flatrate", []) + watch_data.get("free", [])
                            streaming_on = list(dict.fromkeys([p.get("provider_name") for p in stream_list]))
                            
                            # --- NEW: Extract Key Cast and Crew (Token Efficient) ---
                            credits_data = details.get("credits", {})
                            
                            # Grab the top 5 billed actors and their character names
                            top_cast = []
                            for actor in credits_data.get("cast", [])[:5]:
                                top_cast.append(f"{actor.get('name')} (as {actor.get('character')})")
                                
                            # Extract key creative leaders based on media format type
                            key_creators = []
                            if media_type == "movie":
                                # For films, extract the Director(s) from the crew list
                                key_creators = [
                                    crew.get("name") for crew in credits_data.get("crew", [])
                                    if crew.get("job") == "Director"
                                ]
                            else:
                                # For television shows, extract the Creator(s) defined at the root schema
                                key_creators = [creator.get("name") for creator in details.get("created_by", [])]

                            format_label = "Theatrical Film / Movie" if media_type == "movie" else "Television Series / Show"
                            creator_title = "Director(s)" if media_type == "movie" else "Series Creator(s)"
                            
                            return {
                                "source": "The Movie Database (TMDB)",
                                "retrieved_media_format": format_label,
                                "title": title,
                                "release_date": release_date,
                                "genres": genres,
                                creator_title: key_creators if key_creators else "Unknown",
                                "top_billed_cast": top_cast if top_cast else "Unknown",
                                "runtime_or_length": runtime,
                                "vote_average": f"{details.get('vote_average')}/10",
                                "streaming_on": streaming_on if streaming_on else "Not currently streaming for free/subscription in the US.",
                                "overview": details.get("overview"),
                                "system_directive": f"You successfully retrieved data for a {format_label}. If the user explicitly asked for a different format (e.g., they asked for a TV show but you found a Movie), you must inform them of this mismatch."
                            }
                        return {"error": f"TMDB details verification failed with status code: {resp.status}"}
                
                except Exception as e:
                    return {"error": f"An error occurred querying TMDB data structures: {str(e)}"}
