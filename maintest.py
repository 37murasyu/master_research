import utils  # utils.py から関数をインポート

def main():
    name = "Alice"
    greeting = utils.say_hello(name)
    print(greeting)

if __name__ == "__main__":
    main()
