package src.question.type;

import utils.Rand;

public class Audio implements IType {
    private int MAX_LENGTH_NUMBER = 100;
    private int MIN_LENGTH_NUMBER = 5;
    private int length;

    public Audio() {
        createQuestionType();
    }

    private void createQuestionType() {
        Rand rand = new Rand();
        this.length = rand.getInt(MIN_LENGTH_NUMBER, MAX_LENGTH_NUMBER);
    }

    public Integer getQuestionInfo() {
        return length;
    }
}
