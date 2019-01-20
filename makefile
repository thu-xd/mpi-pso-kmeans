#
# makefile
# Dong Xie, 2019-01-16 23:34
#

vpath %.c src

SRC = $(wildcard src/*.c)
SRC_FILE=$(notdir $(SRC))
BUILD_DIR= src/build
OBJ = $(patsubst %.c, $(BUILD_DIR)/%.o,$(SRC_FILE))
DEP = $(patsubst %.c, $(BUILD_DIR)/%.d,$(SRC_FILE))

LDFLAGS= -lm

main : ${OBJ} ${DEP}	
	mpicc -o $@ ${OBJ} $(LDFLAGS)

-include $(DEP)

$(BUILD_DIR)/%.d: %.c
	mpicc $< -MM -MT $(@:.d=.o) >$@

$(BUILD_DIR)/%.o: %.c
	mpicc -o $@ -c $<

.PHONY:clean
clean:
	@rm -f main ${OBJ} ${DEP}
	
# vim:ft=make
#
