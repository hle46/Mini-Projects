CC = clang++
CFLAGS =-c -Wall -Wextra -std=c++11
LDFLAGS =
SRCS = PatternCount.cpp
OBJS = $(SRCS:.cpp=.o)

all: PatternCount

PatternCount: PatternCount.o
	$(CC) $(LDFLAGS) -o $@ $<

%.o: %.cpp
	$(CC) $(CFLAGS) -o $@ $<

clean:
	rm -fr *.o PatternCount 

.PHONY: all clean 
