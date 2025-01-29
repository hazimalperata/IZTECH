package utils;

import java.util.List;
import java.util.Random;

public class Rand<T> {
    Random random = new Random();

    public int getInt(int min_num, int max_num) {
        return random.nextInt((max_num - min_num) + 1) + min_num;
    }

    public int getInt(int max_num){
        return random.nextInt(max_num + 1);
    }

    public T getListObject(List<T> objectList) {
        final int MAX_INDEX = objectList.size() - 1;
        final int MIN_INDEX = 0;
        Random random = new Random();

        int selectedIndex = random.nextInt((MAX_INDEX - MIN_INDEX) + 1) + MIN_INDEX;

        return objectList.get(selectedIndex);
    }
}

