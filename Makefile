CC = gcc
CFLAGS = -O3 -mcpu=apple-m1 -ffast-math -flto -Isrc # -g
LDFLAGS = -lm -framework Accelerate
SRCS = $(wildcard src/*.c)
OUT = out/main

all: $(OUT)

$(OUT): $(SRCS)
	@mkdir -p out
	$(CC) $(CFLAGS) $(SRCS) -o $(OUT) $(LDFLAGS)

debug: 

clean:
	rm -rf out/