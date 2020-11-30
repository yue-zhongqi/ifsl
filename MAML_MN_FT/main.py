from io_utils import parse_args
from tests.MetaTrain import MetaTrain

def func_not_found():  # just in case we dont have the function
    print('No Function ' + ' Found!')


def main():
    params = parse_args('test')
    if params.method == "metatrain":
        tester = MetaTrain(params)
    func = getattr(tester, params.test, func_not_found)
    func()

main()