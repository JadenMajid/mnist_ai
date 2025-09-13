CC=gcc
ODIR=obj

main:
	$(CC) ./src/main.c -o ./out/main.o

clean:
	rm out/*.o

test_linalg: 
	$(CC) ./test/test_linalg.c -o ./out/test_linalg.o
	./out/test_linalg.o

