import random
import json
import torch
import re
import time 
from nltk.stem.porter import PorterStemmer
from model import NeuralNet
from nltk_utils import bag_of_words, tokenize 

BLUE = "\033[94m"
CYAN = "\033[96m"
YELLOW = "\033[93m"
RED = "\033[91m"
RESET = "\033[0m"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

start_load_time = time.time()
with open('intents.json', 'r') as json_data:
    intents = json.load(json_data)

with open('recipes.json', 'r') as json_data1:
    recipes = json.load(json_data1)

FILE = "data.pth"
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

end_load_time = time.time()
load_execution_time = end_load_time - start_load_time
print(f"Model and recipes loaded in {load_execution_time:.4f} seconds.")

stemmer = PorterStemmer()

bot_name = "Yummily"
currentRecipe = None
savedRecipe = []
return_recipe = False
recipe_null = True

def levenshtein_distance(word1, word2):
    len1, len2 = len(word1), len(word2)
    dp = [[0] * (len2 + 1) for _ in range(len1 + 1)]
    
    for i in range(len1 + 1):
        for j in range(len2 + 1):
            if i == 0:
                dp[i][j] = j
            elif j == 0:
                dp[i][j] = i
            elif word1[i - 1] == word2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])

    return dp[len1][len2]


def get_best_match(user_input, possible_tags, threshold=2):
    best_tag = None
    lowest_distance = float('inf')

    for tag in possible_tags:
        distance = levenshtein_distance(user_input, tag)
        if distance < lowest_distance and distance <= threshold:
            lowest_distance = distance
            best_tag = tag

    return best_tag



def check_all_messages(stemmed_words):
    start_process_time = time.time()
    X = bag_of_words(stemmed_words, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    output = model(X)
    _, predicted = torch.max(output, dim=1)
    predicted_tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]

    if prob.item() > 0.75:
        for intent in intents['intents']:
            if predicted_tag == intent["tag"]:
                end_process_time = time.time()
                execution_time = end_process_time - start_process_time
                print(f"Response generated in {execution_time:.4f} seconds.")
                if predicted_tag == "recipe":
                    return suggestRecipe()
                elif predicted_tag in ["allergy", "dislike"]:
                    return removeRecipe(predicted_tag)
                elif predicted_tag == "goodbye":
                    return "EXIT"
                elif predicted_tag == "like":
                    return likeRecipe(predicted_tag)
                elif predicted_tag == "save":
                    return saveRecipe()
                elif predicted_tag == "show":
                    return showRecipe()
                else:
                    return random.choice(intent['responses'])
    else:
        user_input = ' '.join(stemmed_words) 
        best_match_tag = get_best_match(user_input, tags)
        
        end_process_time = time.time()
        execution_time = end_process_time - start_process_time
        print(f"\nResponse generated in {execution_time:.4f} seconds.")

        if best_match_tag:
            for intent in intents['intents']:
                if best_match_tag == intent["tag"]:
                    return random.choice(intent['responses'])
        else:
            return "I do not understand... could you please repeat?"

def saveRecipe():
    if currentRecipe is not None:
        if currentRecipe not in savedRecipe:
            savedRecipe.append(currentRecipe)
            global recipe_null
            recipe_null = False
            return "The recipe has been saved."
        else:
            return "The recipe has already been saved."
    else:
        return "You havent saved any recipe yet"

def showRecipe():
    if recipe_null == False:
        print(CYAN+f"\n{bot_name}: No problem, here is the recipe that has been saved"+RESET)
        return suggestRecipe(filtered_recipes = savedRecipe)
    else:
        return "There is no saved recipe available"

def removeRecipe(tag):
    if tag == "allergy":
        print(CYAN+f"{bot_name}: Seems like you are allergy to certain ingredient, which ingredient are u allergic to?"+RESET)
    elif tag == "dislike":
        print(CYAN+f"{bot_name}: Seems like you dont wish to include certain ingredient, which ingredient that you wish to remove? "+RESET)

    remove_input = input("You: ").lower()

    remove_ingredients = tokenize(remove_input)

    recipes_to_remove = []
    for recipe in recipes:
        ingredient_parts = [re.split(r'\s+|[,;?!.-]\s*', ingredient.lower()) for ingredient in recipe["ingredients"]]
        ingredient_words = [word for sublist in ingredient_parts for word in sublist]
        
        for word in remove_ingredients:
            if word in ingredient_words:
                recipes_to_remove.append(recipe)
                break

    for recipe in recipes_to_remove:
        recipes.remove(recipe)

    if not recipes:
        return f"Sorry, no recipes are available without '{remove_input}'."

    print(CYAN+f"{bot_name}: Removed recipes with '{remove_input}' successfully.\n\n"+RESET)
    return suggestRecipe()

def likeRecipe(tag):
    if tag == "like":
        print(CYAN+f"{bot_name}: Seems like you are interested to certain ingredient, which ingredient are you interested in?"+RESET)

    like_input = input("You: ").lower()

    like_ingredients = tokenize(like_input)

    recipes_to_add = []
    for recipe in recipes:
        ingredient_parts = [re.split(r'\s+|[,;?!.-]\s*', ingredient.lower()) for ingredient in recipe["ingredients"]]
        ingredient_words = [word for sublist in ingredient_parts for word in sublist]
        
        for word in like_ingredients:
            if word in ingredient_words:
                recipes_to_add.append(recipe)
                break

    if not recipes:
        return f"Sorry, no recipes are available with '{like_input}'."

    print(f"Added recipes with '{like_input}' successfully.\n\n")
    return suggestRecipe(filtered_recipes=recipes_to_add)

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
        print(CYAN+"\nHere are the top-selected recipes:"+RESET)
        for i, recipe in enumerate(batch_recipes):
            print(f"{i + 1}. {recipe['name']} (Votes: {recipe['vote_count']})")

        print(CYAN+f"\n{bot_name}: Which recipe would you like to choose from 1 - X ('next' to reshuffle, 'back' to exit): "+RESET)
        choice = input(YELLOW+"You: "+RESET)
        global start_time
        start_time = time.time()

        if choice.lower() == "next":
            current_batch += 1
            return suggestRecipe(filtered_recipes, current_batch, suggested_recipes)
        elif choice.lower() == "back":
            print(CYAN+f"\n{bot_name}: I will stop suggesting then."+RESET)
            return None
        else:
            try:
                index = int(choice) - 1
                if 0 <= index < len(batch_recipes):
                    return displayRecipe(batch_recipes[index], start_time)
                else:
                    print("Invalid choice. Please enter a number from 1 - X.")
            except ValueError:
                print("I don't understand that...")


def displayRecipe(recipe, start_time):
    end_time = time.time()  
    elapsed_time = end_time - start_time  
    print(CYAN+f"{bot_name}: Here you go ~"+RESET)
    print("=" * 150)
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


# Main loop
def main():
    
    while True:
        global start_time
        start_time = time.time()
        user_input = input(YELLOW+"You: "+RESET)

        stemmed_words = [stemmer.stem(w) for w in tokenize(user_input)]
        response = check_all_messages(stemmed_words)

        if user_input.lower() == "quit" or response== "EXIT":
            print(CYAN+f"{bot_name}: Thanks for using Yummily Foodbot and hope to help you again!"+RESET)
            break
        
        elif response is not None:
            print(CYAN+f"\n{bot_name}: {response}"+RESET)
