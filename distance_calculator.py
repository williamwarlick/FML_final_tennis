from geopy.geocoders import Nominatim
from geopy.distance import great_circle

def get_coordinates(location):
    geolocator = Nominatim(user_agent="geoapiExercises")
    loc = geolocator.geocode(location)
    return (loc.latitude, loc.longitude)

def main():
    country = "France"  # Replace with the name of the country
    city = "New York City, USA"  # Replace with the name of the city and its country

    country_coordinates = get_coordinates(country)
    city_coordinates = get_coordinates(city)

    distance = great_circle(country_coordinates, city_coordinates).km

    print(f"The distance between {country} and {city} is approximately {distance:.2f} km.")


if __name__ == '__main__':
    main()