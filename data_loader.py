import pandas as pd
import numpy as np

# Curated dataset of iconic & diverse movies with rich metadata and reliable poster art
MOVIES_DATA = [
    {
        "movie_id": 1,
        "title": "Inception",
        "release_year": 2010,
        "genres": "Sci-Fi, Action, Thriller",
        "director": "Christopher Nolan",
        "cast": "Leonardo DiCaprio, Joseph Gordon-Levitt, Elliot Page, Tom Hardy",
        "plot_synopsis": "A skilled thief who steals corporate secrets through dream-sharing technology is given the inverse task of planting an idea into the mind of a CEO.",
        "vote_average": 8.8,
        "vote_count": 34000,
        "runtime_min": 148,
        "poster_url": "https://images.unsplash.com/photo-1534447677768-be436bb09401?w=500&auto=format&fit=crop&q=80",
        "tags": "dream subconscious heist mind-bending reality layered architecture"
    },
    {
        "movie_id": 2,
        "title": "Interstellar",
        "release_year": 2014,
        "genres": "Sci-Fi, Drama, Adventure",
        "director": "Christopher Nolan",
        "cast": "Matthew McConaughey, Anne Hathaway, Jessica Chastain, Michael Caine",
        "plot_synopsis": "When Earth becomes uninhabitable, a team of ex-NASA astronauts travels through a wormhole near Saturn in search of a new home for humanity.",
        "vote_average": 8.7,
        "vote_count": 31000,
        "runtime_min": 169,
        "poster_url": "https://images.unsplash.com/photo-1451187580459-43490279c0fa?w=500&auto=format&fit=crop&q=80",
        "tags": "space exploration wormhole time dilation relativity black hole love survival"
    },
    {
        "movie_id": 3,
        "title": "The Dark Knight",
        "release_year": 2008,
        "genres": "Action, Crime, Drama",
        "director": "Christopher Nolan",
        "cast": "Christian Bale, Heath Ledger, Aaron Eckhart, Gary Oldman",
        "plot_synopsis": "Batman battles the psychotic criminal mastermind known as the Joker, who plunges Gotham City into chaos while testing the limits of morality.",
        "vote_average": 9.0,
        "vote_count": 42000,
        "runtime_min": 152,
        "poster_url": "https://images.unsplash.com/photo-1509198397868-475647b2a1e5?w=500&auto=format&fit=crop&q=80",
        "tags": "batman joker gotham vigilantism chaos justice morality crime thriller"
    },
    {
        "movie_id": 4,
        "title": "The Matrix",
        "release_year": 1999,
        "genres": "Sci-Fi, Action",
        "director": "Lana Wachowski, Lilly Wachowski",
        "cast": "Keanu Reeves, Laurence Fishburne, Carrie-Anne Moss, Hugo Weaving",
        "plot_synopsis": "A computer hacker discovers that shocking reality is an artificial simulation controlled by sentient machines, and joins a rebellion to free mankind.",
        "vote_average": 8.7,
        "vote_count": 29000,
        "runtime_min": 136,
        "poster_url": "https://images.unsplash.com/photo-1526374965328-7f61d4dc18c5?w=500&auto=format&fit=crop&q=80",
        "tags": "simulation cyberpunk martial arts ai rebellion red pill virtual reality"
    },
    {
        "movie_id": 5,
        "title": "Shutter Island",
        "release_year": 2010,
        "genres": "Mystery, Thriller, Drama",
        "director": "Martin Scorsese",
        "cast": "Leonardo DiCaprio, Mark Ruffalo, Ben Kingsley, Michelle Williams",
        "plot_synopsis": "Two U.S. Marshals are sent to an isolated mental institution on a remote island to investigate the mysterious disappearance of a patient.",
        "vote_average": 8.2,
        "vote_count": 22000,
        "runtime_min": 138,
        "poster_url": "https://images.unsplash.com/photo-1518709268805-4e9042af9f23?w=500&auto=format&fit=crop&q=80",
        "tags": "psychological thriller asylum plot twist hallucination detective trauma storm"
    },
    {
        "movie_id": 6,
        "title": "Pulp Fiction",
        "release_year": 1994,
        "genres": "Crime, Drama",
        "director": "Quentin Tarantino",
        "cast": "John Travolta, Samuel L. Jackson, Uma Thurman, Bruce Willis",
        "plot_synopsis": "The lives of two mob hitmen, a boxer, a gangster and his wife, and a pair of diner bandits intertwine in four tales of violence and redemption.",
        "vote_average": 8.9,
        "vote_count": 35000,
        "runtime_min": 154,
        "poster_url": "https://images.unsplash.com/photo-1594909122845-11baa439b7bf?w=500&auto=format&fit=crop&q=80",
        "tags": "nonlinear dialogue mobsters hitman dark comedy cult classic dance adrenaline"
    },
    {
        "movie_id": 7,
        "title": "Goodfellas",
        "release_year": 1990,
        "genres": "Biography, Crime, Drama",
        "director": "Martin Scorsese",
        "cast": "Robert De Niro, Ray Liotta, Joe Pesci, Lorraine Bracco",
        "plot_synopsis": "The rise and fall of mob associate Henry Hill and his friends through the Italian-American crime syndicate of New York over three decades.",
        "vote_average": 8.7,
        "vote_count": 20000,
        "runtime_min": 145,
        "poster_url": "https://images.unsplash.com/photo-1542751371-adc38448a05e?w=500&auto=format&fit=crop&q=80",
        "tags": "mafia organized crime brooklyn heist betrayal drugs rise and fall"
    },
    {
        "movie_id": 8,
        "title": "The Godfather",
        "release_year": 1972,
        "genres": "Crime, Drama",
        "director": "Francis Ford Coppola",
        "cast": "Marlon Brando, Al Pacino, James Caan, Robert Duvall",
        "plot_synopsis": "The aging patriarch of an organized crime dynasty in postwar New York transfers control of his clandestine empire to his reluctant youngest son.",
        "vote_average": 9.2,
        "vote_count": 38000,
        "runtime_min": 175,
        "poster_url": "https://images.unsplash.com/photo-1512453979798-5ea266f8880c?w=500&auto=format&fit=crop&q=80",
        "tags": "corleone family mafia succession loyalty power italian syndicate legacy"
    },
    {
        "movie_id": 9,
        "title": "Spirited Away",
        "release_year": 2001,
        "genres": "Animation, Adventure, Family, Fantasy",
        "director": "Hayao Miyazaki",
        "cast": "Rumi Hiiragi, Miyu Irino, Mari Natsuki, Takashi Naito",
        "plot_synopsis": "During her family's move to the suburbs, a sullen 10-year-old girl wanders into a magical bathhouse world ruled by spirits, witches, and monsters.",
        "vote_average": 8.6,
        "vote_count": 18000,
        "runtime_min": 125,
        "poster_url": "https://images.unsplash.com/photo-1578632767115-351597cf2477?w=500&auto=format&fit=crop&q=80",
        "tags": "ghibli anime spirits bathhouse fantasy magic dragon coming of age"
    },
    {
        "movie_id": 10,
        "title": "Spider-Man: Into the Spider-Verse",
        "release_year": 2018,
        "genres": "Animation, Action, Adventure, Sci-Fi",
        "director": "Bob Persichetti, Peter Ramsey, Rodney Rothman",
        "cast": "Shameik Moore, Jake Johnson, Hailee Steinfeld, Mahershala Ali",
        "plot_synopsis": "Teenager Miles Morales becomes the new Spider-Man and must join with five alternate-dimension spider-heroes to stop a kingpin threat to all realities.",
        "vote_average": 8.4,
        "vote_count": 21000,
        "runtime_min": 117,
        "poster_url": "https://images.unsplash.com/photo-1635805737707-575885ab0820?w=500&auto=format&fit=crop&q=80",
        "tags": "multiverse comic book animation hip hop superhero coming of age brooklyn"
    },
    {
        "movie_id": 11,
        "title": "Parasite",
        "release_year": 2019,
        "genres": "Drama, Thriller, Comedy",
        "director": "Bong Joon Ho",
        "cast": "Song Kang-ho, Lee Sun-kyun, Cho Yeo-jeong, Choi Woo-shik",
        "plot_synopsis": "Greed and class discrimination threaten the newly formed symbiotic relationship between the wealthy Park family and the destitute Kim clan.",
        "vote_average": 8.5,
        "vote_count": 25000,
        "runtime_min": 132,
        "poster_url": "https://images.unsplash.com/photo-1536440136628-849c177e76a1?w=500&auto=format&fit=crop&q=80",
        "tags": "class divide dark comedy deception architecture basement south korea tension"
    },
    {
        "movie_id": 12,
        "title": "Whiplash",
        "release_year": 2014,
        "genres": "Drama, Music",
        "director": "Damien Chazelle",
        "cast": "Miles Teller, J.K. Simmons, Paul Reiser, Melissa Benoist",
        "plot_synopsis": "A promising young drummer enrolls at a cut-throat music conservatory where his ruthless instructor pushes him past psychological breaking points.",
        "vote_average": 8.5,
        "vote_count": 19000,
        "runtime_min": 106,
        "poster_url": "https://images.unsplash.com/photo-1511671782779-c97d3d27a1d4?w=500&auto=format&fit=crop&q=80",
        "tags": "jazz drums obsession perfection mentor abuse intense tempo passion"
    },
    {
        "movie_id": 13,
        "title": "Blade Runner 2049",
        "release_year": 2017,
        "genres": "Sci-Fi, Mystery, Action",
        "director": "Denis Villeneuve",
        "cast": "Ryan Gosling, Harrison Ford, Ana de Armas, Sylvia Hoeks",
        "plot_synopsis": "Young Blade Runner K unearths a long-buried secret that has the potential to plunge what's left of society into chaos, leading him to track down Rick Deckard.",
        "vote_average": 8.0,
        "vote_count": 16000,
        "runtime_min": 164,
        "poster_url": "https://images.unsplash.com/photo-1508739773434-c26b3d09e071?w=500&auto=format&fit=crop&q=80",
        "tags": "cyberpunk replicant neo-noir dystopia hologram memory identity existential"
    },
    {
        "movie_id": 14,
        "title": "Dune: Part One",
        "release_year": 2021,
        "genres": "Sci-Fi, Adventure, Action",
        "director": "Denis Villeneuve",
        "cast": "Timothée Chalamet, Rebecca Ferguson, Oscar Isaac, Zendaya",
        "plot_synopsis": "Paul Atreides leads nomadic tribes in a battle to control the desert planet Arrakis, the galaxy's sole source of the universe's most precious resource: spice.",
        "vote_average": 8.0,
        "vote_count": 22000,
        "runtime_min": 155,
        "poster_url": "https://images.unsplash.com/photo-1509316975850-ff9c5deb0cd9?w=500&auto=format&fit=crop&q=80",
        "tags": "desert sand sandworm spice empire prophecy messiah political intrigue"
    },
    {
        "movie_id": 15,
        "title": "The Grand Budapest Hotel",
        "release_year": 2014,
        "genres": "Comedy, Adventure, Crime",
        "director": "Wes Anderson",
        "cast": "Ralph Fiennes, F. Murray Abraham, Mathieu Amalric, Adrien Brody",
        "plot_synopsis": "A writer encounters the owner of an aging high-class European hotel, who recounts his years as a legendary concierge in the midst of a violent inheritance war.",
        "vote_average": 8.1,
        "vote_count": 17000,
        "runtime_min": 99,
        "poster_url": "https://images.unsplash.com/photo-1566073771259-6a8506099945?w=500&auto=format&fit=crop&q=80",
        "tags": "symmetry pastel concierge painting theft quirky caper europe vintage"
    },
    {
        "movie_id": 16,
        "title": "Everything Everywhere All at Once",
        "release_year": 2022,
        "genres": "Action, Adventure, Comedy, Sci-Fi",
        "director": "Daniel Kwan, Daniel Scheinert",
        "cast": "Michelle Yeoh, Stephanie Hsu, Ke Huy Quan, Jamie Lee Curtis",
        "plot_synopsis": "A middle-aged Chinese immigrant is swept into an insane adventure in which she alone can save existence by exploring other universes and connecting with the lives she could have led.",
        "vote_average": 7.8,
        "vote_count": 15000,
        "runtime_min": 139,
        "poster_url": "https://images.unsplash.com/photo-1518770660439-4636190af475?w=500&auto=format&fit=crop&q=80",
        "tags": "multiverse bagel googly eyes family nihilism kung fu mother daughter absurdist"
    },
    {
        "movie_id": 17,
        "title": "Coco",
        "release_year": 2017,
        "genres": "Animation, Adventure, Family, Fantasy",
        "director": "Lee Unkrich, Adrian Molina",
        "cast": "Anthony Gonzalez, Gael García Bernal, Benjamin Bratt, Alanna Ubach",
        "plot_synopsis": "Aspiring musician Miguel, confronted with his family's ancestral ban on music, enters the Land of the Dead to find his great-great-grandfather, a legendary singer.",
        "vote_average": 8.4,
        "vote_count": 19000,
        "runtime_min": 105,
        "poster_url": "https://images.unsplash.com/photo-1514525253161-7a46d19cd819?w=500&auto=format&fit=crop&q=80",
        "tags": "day of the dead skeleton guitar family memory mexico music afterlife"
    },
    {
        "movie_id": 18,
        "title": "Knives Out",
        "release_year": 2019,
        "genres": "Comedy, Crime, Drama, Mystery",
        "director": "Rian Johnson",
        "cast": "Daniel Craig, Chris Evans, Ana de Armas, Jamie Lee Curtis",
        "plot_synopsis": "A detective investigates the death of a patriarch of an eccentric, combative family after an 85th birthday celebration turns lethal.",
        "vote_average": 7.9,
        "vote_count": 18000,
        "runtime_min": 130,
        "poster_url": "https://images.unsplash.com/photo-1544717305-2782549b5136?w=500&auto=format&fit=crop&q=80",
        "tags": "whodunnit detective mansion inheritance will poison eccentric family humor"
    },
    {
        "movie_id": 19,
        "title": "Avengers: Endgame",
        "release_year": 2019,
        "genres": "Action, Adventure, Drama, Sci-Fi",
        "director": "Anthony Russo, Joe Russo",
        "cast": "Robert Downey Jr., Chris Evans, Mark Ruffalo, Chris Hemsworth",
        "plot_synopsis": "After the devastating events of Infinity War, the universe is in ruins. With the help of remaining allies, the Avengers assemble once more to reverse Thanos' actions.",
        "vote_average": 8.4,
        "vote_count": 27000,
        "runtime_min": 181,
        "poster_url": "https://images.unsplash.com/photo-1607604276583-eef5d076aa5f?w=500&auto=format&fit=crop&q=80",
        "tags": "marvel superheroes time heist thanos infinity stones finale sacrifice iron man"
    },
    {
        "movie_id": 20,
        "title": "The Truman Show",
        "release_year": 1998,
        "genres": "Comedy, Drama, Sci-Fi",
        "director": "Peter Weir",
        "cast": "Jim Carrey, Laura Linney, Ed Harris, Noah Emmerich",
        "plot_synopsis": "An insurance salesman discovers his entire quiet life is actually an elaborate television broadcast recorded 24/7 and watched by millions around the globe.",
        "vote_average": 8.2,
        "vote_count": 21000,
        "runtime_min": 103,
        "poster_url": "https://images.unsplash.com/photo-1485846234645-a62644f84728?w=500&auto=format&fit=crop&q=80",
        "tags": "reality tv fake world surveillance escape existential paranoia dome sailboat"
    }
]

# User personas with distinct taste profiles for Collaborative Filtering:
# User 1: "Sci-Fi & Mind-Benders Enthusiast" (Inception, Matrix, Interstellar, Blade Runner)
# User 2: "Mob & Crime Drama Aficionado" (Godfather, Goodfellas, Pulp Fiction, Dark Knight)
# User 3: "Family & Vibrant Animation Fan" (Spirited Away, Spider-Verse, Coco)
# User 4: "Eclectic Cinephile & Dark Humor" (Parasite, Grand Budapest, Everything Everywhere, Knives Out)
# User 5: "Cold-Start User" (Only 1 rating, triggers hybrid cold-start logic)
RATINGS_DATA = [
    # User 1: Sci-Fi / Mind-Bender
    {"user_id": 1, "movie_id": 1, "rating": 5.0, "watch_ratio": 1.0},
    {"user_id": 1, "movie_id": 2, "rating": 5.0, "watch_ratio": 1.0},
    {"user_id": 1, "movie_id": 4, "rating": 4.5, "watch_ratio": 0.95},
    {"user_id": 1, "movie_id": 13, "rating": 4.5, "watch_ratio": 0.9},
    {"user_id": 1, "movie_id": 14, "rating": 4.0, "watch_ratio": 0.85},
    {"user_id": 1, "movie_id": 9, "rating": 2.0, "watch_ratio": 0.3},

    # User 2: Crime & Noir
    {"user_id": 2, "movie_id": 6, "rating": 5.0, "watch_ratio": 1.0},
    {"user_id": 2, "movie_id": 7, "rating": 5.0, "watch_ratio": 1.0},
    {"user_id": 2, "movie_id": 8, "rating": 5.0, "watch_ratio": 1.0},
    {"user_id": 2, "movie_id": 3, "rating": 4.5, "watch_ratio": 0.9},
    {"user_id": 2, "movie_id": 5, "rating": 4.0, "watch_ratio": 0.85},
    {"user_id": 2, "movie_id": 17, "rating": 1.5, "watch_ratio": 0.2},

    # User 3: Animation & Wholesome
    {"user_id": 3, "movie_id": 9, "rating": 5.0, "watch_ratio": 1.0},
    {"user_id": 3, "movie_id": 10, "rating": 5.0, "watch_ratio": 1.0},
    {"user_id": 3, "movie_id": 17, "rating": 5.0, "watch_ratio": 1.0},
    {"user_id": 3, "movie_id": 15, "rating": 4.0, "watch_ratio": 0.85},
    {"user_id": 3, "movie_id": 7, "rating": 1.0, "watch_ratio": 0.15},

    # User 4: Eclectic Auteur & Dark Comedy
    {"user_id": 4, "movie_id": 11, "rating": 5.0, "watch_ratio": 1.0},
    {"user_id": 4, "movie_id": 12, "rating": 4.5, "watch_ratio": 0.95},
    {"user_id": 4, "movie_id": 15, "rating": 4.5, "watch_ratio": 0.9},
    {"user_id": 4, "movie_id": 16, "rating": 5.0, "watch_ratio": 1.0},
    {"user_id": 4, "movie_id": 18, "rating": 4.0, "watch_ratio": 0.8},

    # User 5: Cold-Start (New user)
    {"user_id": 5, "movie_id": 1, "rating": 5.0, "watch_ratio": 1.0},

    # User 6: Christopher Nolan / Epic action
    {"user_id": 6, "movie_id": 1, "rating": 5.0, "watch_ratio": 1.0},
    {"user_id": 6, "movie_id": 2, "rating": 5.0, "watch_ratio": 1.0},
    {"user_id": 6, "movie_id": 3, "rating": 5.0, "watch_ratio": 1.0},
    {"user_id": 6, "movie_id": 19, "rating": 4.5, "watch_ratio": 0.95},

    # User 7: Psychological Mystery & Mind Games
    {"user_id": 7, "movie_id": 1, "rating": 4.5, "watch_ratio": 0.9},
    {"user_id": 7, "movie_id": 5, "rating": 5.0, "watch_ratio": 1.0},
    {"user_id": 7, "movie_id": 18, "rating": 4.5, "watch_ratio": 0.9},
    {"user_id": 7, "movie_id": 20, "rating": 4.5, "watch_ratio": 0.9},
    {"user_id": 7, "movie_id": 11, "rating": 4.0, "watch_ratio": 0.85},
]

def load_or_create_datasets():
    df_movies = pd.DataFrame(MOVIES_DATA)
    df_ratings = pd.DataFrame(RATINGS_DATA)
    df_movies.to_csv("movies.csv", index=False)
    df_ratings.to_csv("ratings.csv", index=False)
    print(f"Loaded {len(df_movies)} movies and {len(df_ratings)} ratings successfully.")
    return df_movies, df_ratings

if __name__ == "__main__":
    load_or_create_datasets()
