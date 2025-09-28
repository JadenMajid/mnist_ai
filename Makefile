CC=gcc
ODIR=out

main:
	$(CC) ./src/main.c -o ./$(ODIR)/main.o

clean:
	rm $(ODIR)/*.o

build_test:
	$(CC) ./test/test_linalg.c -o ./$(ODIR)/test_linalg.o

test: test_linalg

test_leaks: build_test
	valgrind --leak-check=full --show-leak-kinds=all ./$(ODIR)/test_linalg.o

test_linalg: build_test
	./$(ODIR)/test_linalg.o

