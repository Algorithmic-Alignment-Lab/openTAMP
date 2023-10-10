if __name__ == "__main__":
    x = 3
    import os
    mas, slav = os.openpty()
    print(os.isatty(slav))
    import multiprocessing as mp
    mp.freeze_support()
    mp.set_start_method('spawn')
    input('Press any key to continue!')
    import pdb; pdb.set_trace()


