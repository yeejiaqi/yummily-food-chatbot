import re
import nltk
import json
import random
import time
from nltk.stem import WordNetLemmatizer

nltk.download('wordnet')
nltk.download('omw-1.4')

BLUE = "\033[94m"
CYAN = "\033[96m"
YELLOW = "\033[93m"
RED = "\033[91m"
RESET = "\033[0m"

lemmatizer = WordNetLemmatizer()

bot_name = "Yummily"
currentRecipe = None
savedRecipe = []
return_recipe = False
recipe_null = True

recipes_file = open("recipes.json","r")
recipes_data = recipes_file.read()
recipes = json.loads(recipes_data)

response_file = open("response.json","r")
response_data = response_file.read()
responses = json.loads(response_data)

def message_probability(user_message, recognised_words, single_response=False, required_words=[]):
    message_certainty = 0
    has_required_words = True

    for word in user_message:
        if word in recognised_words:
            message_certainty += 1

    percentage = float(message_certainty) / float(len(recognised_words))

    for word in required_words:
        if word not in user_message:
            has_required_words = False
            break

    if has_required_words or single_response:
        return int(percentage * 100)
    else:
        return 0

def unknown():
    response = ["Could you please re-phrase that?",
                "...",
                "Sounds about right",
                "What does that mean?"][random.randrange(4)]
    return response

def check_all_messages(message):
    highest_prob_list = {}

    def response(bot_response, list_of_words, single_response=False, required_words=[]):
        nonlocal highest_prob_list
        chosen_response = bot_response[0] if len(bot_response) == 1 else random.choice(bot_response)
        highest_prob_list[chosen_response] = message_probability(message, list_of_words, single_response, required_words)

    for entry in responses:
        response(entry["bot_response"], entry["list of words"], entry["single_response"], entry["required_words"])
    

    best_match = max(highest_prob_list, key=highest_prob_list.get)
    
    return unknown() if highest_prob_list[best_match] < 1 else best_match

def get_response(user_input):
    split_message = re.split(r'\s+|[,;?!.-]\s*', user_input.lower())
    lemmas = [lemmatizer.lemmatize(word) for word in split_message]
    response = check_all_messages(lemmas)
    return response

def suggestRecipe(filtered_recipes=None, current_batch=0, suggested_recipes=None):
    if filtered_recipes is None:
        filtered_recipes = recipes

    if suggested_recipes is None:
        suggested_recipes = set()  

    sorted_recipes = sorted(filtered_recipes, key=lambda recipe: recipe["vote_count"], reverse=True)

    filtered_sorted_recipes = [recipe for recipe in sorted_recipes if recipe["name"] not in suggested_recipes]

    batch_size = 5
    start_index = current_batch * batch_size
    end_index = start_index + batch_size

    batch_recipes = []
    unique_recipe_names = set()  

    for recipe in filtered_sorted_recipes[start_index:end_index]:
        if recipe["name"] not in unique_recipe_names: 
            batch_recipes.append(recipe)
            unique_recipe_names.add(recipe["name"])

        if len(batch_recipes) == batch_size:
            break  

    if not batch_recipes:
        if current_batch == 0:
            return "No more recipes available."
        else:
            return suggestRecipe(filtered_recipes, current_batch=0, suggested_recipes=suggested_recipes)

    for recipe in batch_recipes:
        suggested_recipes.add(recipe["name"])

    while True:  
        print(CYAN+"\nTop-selected recipes:"+RESET)
        for i, recipe in enumerate(batch_recipes):
            print(f"{i + 1}. {recipe['name']} (Votes: {recipe['vote_count']})")

        print(CYAN+f"\n{bot_name}: Which recipe would you like to choose from 1 - X ('next' to reshuffle, 'back' to exit): ")
        choice = input(YELLOW+"You: "+RESET)

        if choice.lower() == "next":
            current_batch += 1
            return suggestRecipe(filtered_recipes, current_batch, suggested_recipes)
        elif choice.lower() == "back":
            print(CYAN+f"{bot_name}: I will stop suggesting then."+RESET)
            return None
        else:
            try:
                index = int(choice) - 1
                if 0 <= index < len(batch_recipes):
                    global return_recipe
                    return_recipe = True
                    return displayRecipe(batch_recipes[index])
                else:
                    print("Invalid choice. Please enter a number from 1 - x.")
            except ValueError:
                print("I don't understand that...")

def displayRecipe(recipe):
    print(CYAN+f"{bot_name}: Here you go ~"+RESET)
    print("\n===========================================================================================================================")
    print(BLUE+f"Name\t\t: {recipe['name']}"+RESET)
    print(BLUE+f"Description\t: {recipe['description']}"+RESET)
    print(BLUE+f"Rating\t\t: {recipe['rattings']}\n"+RESET)
    print(BLUE + "*****Ingredients*****"+RESET)
    for ingredient in recipe["ingredients"]:
        print(f"- {ingredient}")
    print(BLUE+"\n*****Steps*****"+RESET)
    for idx, step in enumerate(recipe["steps"], start=1):
        print(f"{idx}. {step}")
    print(BLUE+"\n*****Nutrients*****"+RESET)
    print(f". Kcal\t\t: {recipe['nutrients'].get('kcal', 'Not available')}")
    print(f". Fat\t\t: {recipe['nutrients'].get('fat', 'Not available')}")
    print(f". Protein\t: {recipe['nutrients'].get('protein', 'Not available')}")
    print(BLUE+"\n*****Times*****"+RESET)
    print(f". Preparation\t: {recipe['times'].get('Preparation', 'Not available')}")
    print(f". Cooking\t: {recipe['times'].get('Cooking', 'Not available')}")
    print("===========================================================================================================================")
    global currentRecipe
    currentRecipe = recipe
    return "Have fun preparing your dish!"

def triggerFunction(response):
    if response == "The recipe has been saved":
        if currentRecipe is not None:
            if currentRecipe not in savedRecipe:
                savedRecipe.append(currentRecipe)
                global recipe_null
                recipe_null = False
                return response
            else:
                return "The recipe has already been saved."
        else:
            return "You haven't view any recipe yet"
    elif response == "No problem, here is the recipe that has been saved":
        if recipe_null == False:
            print(CYAN+f"{bot_name}: " + response +RESET)
            return suggestRecipe(filtered_recipes = savedRecipe)
        else:
            return "There is no saved recipe available"
    else:
        category_keyword = ["like", "remove", "allergic", "recipes"]
        response_word = re.split(r'\s+|[,;?!.-]\s*', response.lower())

        cond = False

        for word in response_word:
            if word in category_keyword:
                cond = True

        if cond:
            print(CYAN+f"{bot_name}: " + response+RESET)

            if response == "Seems like you dont wish to include certain ingredient, which ingredient that you wish to remove?" or response == "Seems like you are allergy to certain ingredient, which ingredient are u allergic to?":
                user_input = input(YELLOW+"You: "+RESET)
                user_input = re.split(r'\s+|[,;?!.-]\s*', user_input.lower())
                
                for recipe in recipes:
                    ingredient_parts = [re.split(r'\s+|[,;?!.-]\s*', ingredient.lower()) for ingredient in recipe["ingredients"]]
                    ingredient_words = [word for sublist in ingredient_parts for word in sublist]
                    for word in user_input:
                        if word in ingredient_words:
                            recipes.remove(recipe)
                            break
                if not recipes:
                    return f"Sorry, no recipes are available without '{user_input}'."
                
                print(CYAN+f"{bot_name}: Removed recipes with '{user_input}' successfully.\n\n")
                return suggestRecipe()

            elif response == "Seems like you are interested to certain ingredient, which ingredient are you interested in?":
                user_input = input(YELLOW+"You: "+RESET)
                user_input = re.split(r'\s+|[,;?!.-]\s*', user_input.lower())
                
                like = []
                for recipe in recipes:
                    ingredient_parts = [re.split(r'\s+|[,;?!.-]\s*', ingredient.lower()) for ingredient in recipe["ingredients"]]
                    ingredient_words = [word for sublist in ingredient_parts for word in sublist]
                    for word in user_input:
                        if word in ingredient_words:
                            like.append(recipe)
                            break
                if not recipes:
                    return f"Sorry, no recipes are available with '{user_input}'."
                
                print(f"Added recipes with '{user_input}' successfully.\n\n")
                return suggestRecipe(filtered_recipes = like)
            elif response == "Here are some recipes for you!":
                return suggestRecipe()
        else:
            return response

def main():
    while True:
        user_input = input(YELLOW+"You: "+RESET)
        start_time = time.time_ns()

        response = get_response(user_input)

        if user_input.lower() == "quit" or response == "Thanks for using Yummily Foodbot and hope to help you again!":
            print(CYAN+"\nThanks for using Yummily Foodbot and hope to help you again!"+RESET)
            break

        response = triggerFunction(response)

        global return_recipe

        if return_recipe:
            return_recipe = False
        else:
            if response is not None: 
                print(CYAN+f"{bot_name}: " + response+RESET)
                end_time = time.time_ns()
                used_time = end_time - start_time
                used_time_sec = used_time / 1_000_000_000
                print(f"\nThe total time used is {used_time_sec:.8f} seconds")


