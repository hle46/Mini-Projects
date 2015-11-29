CC = clang++
CFLAGS =-c -Wall -Wextra -std=c++11
LDFLAGS =
SRCS = UF.cpp
OBJS = $(SRCS:.cpp=.o)

all: UF

UF: UF.o
	$(CC) $(LDFLAGS) -o $@ $<

%.o: %.cpp
	$(CC) $(CFLAGS) -o $@ $<

clean:
	rm -fr *.o UF 

.PHONY: all clean
