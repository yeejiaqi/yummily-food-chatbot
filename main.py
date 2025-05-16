import RuleBasedChatbot as RBC
import chatbot_SVM as SVM
import Neural_chatbot as NN
import time

bot_name = "Yummily"

start_load_time = time.time()
end_load_time = time.time()
load_execution_time = end_load_time - start_load_time

BLUE = "\033[94m"
CYAN = "\033[96m"
YELLOW = "\033[93m"
RED = "\033[91m"
RESET = "\033[0m"

def main():
    print("\n" * 20)
    while True:
        print("                           Model Selection                  ")
        print("===================================================================")
        print("     Option                   Function")
        print("===================================================================")
        print("       1.                     Rule-Based Chatbot")
        print("       2.                     Neural Network Based Chatbot")
        print("       3.                     SVM Based Chatbot")
        print("       0.                     Exit Program")
        print("===================================================================\n")
        user_input = input("Enter your selection: ")

        if user_input == "1":
            print(BLUE +"\n\n\n\n\n\n\n\n                         *********************************")
            print("                         *          Yummily Chatbot      *")
            print("                         *********************************"+ RESET)
            print(">> type 'quit' to exit")
            print(CYAN +f"\n{bot_name}: Welcome to Yummily Foodbot! We provide over 100+ recipes collected!"+RESET)
            print(CYAN +f"{bot_name}: We provide over 100+ recipes collected!"+RESET)
            print(CYAN +f"{bot_name}: Feel free to tell us what you want"+RESET)
            RBC.main()

        elif user_input == "2":
            print(f"Model and recipes loaded in {load_execution_time:.4f} seconds.")
            print(BLUE +"\n\n\n\n\n\n\n\n                         *********************************")
            print("                         *          Yummily Chatbot      *")
            print("                         *********************************"+ RESET)
            print(">> type 'quit' to exit")
            print(CYAN +f"\n{bot_name}: Welcome to Yummily Foodbot! We provide over 100+ recipes collected!"+RESET)
            print(CYAN +f"{bot_name}: We provide over 100+ recipes collected!"+RESET)
            print(CYAN +f"{bot_name}: Feel free to tell us what you want"+RESET)
            NN.main()

        elif user_input == "3":
            print(BLUE +"\n\n\n\n\n\n\n\n                         *********************************")
            print("                         *          Yummily Chatbot      *")
            print("                         *********************************"+ RESET)
            print(">> type 'quit' to exit")
            print(CYAN +f"\n{bot_name}: Welcome to Yummily Foodbot! We provide over 100+ recipes collected!"+RESET)
            print(CYAN +f"{bot_name}: We provide over 100+ recipes collected!"+RESET)
            print(CYAN +f"{bot_name}: Feel free to tell us what you want"+RESET)
            SVM.main()
        elif user_input == "0":
            print(CYAN +f"\n{bot_name}: Thanks for using Yummily Foodbot. Hope to see you again!"+RESET)
            break
        else:
            print(RED +"Invalid choice, please enter interger in the range of 0-3.\n"+RESET)
        
main()