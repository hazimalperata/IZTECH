package src.question.type;

import utils.Rand;

public class Text implements IType {
    private int MAX_LENGTH_NUMBER = 15;
    private int MIN_LENGTH_NUMBER = 2;
    private String text;

    public Text() {
        createQuestionType();
    }

    private void createQuestionType() {
        String alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnrstuvwxyz";
        StringBuilder sb = new StringBuilder();
        Rand rand = new Rand();
        int length = rand.getInt(MIN_LENGTH_NUMBER, MAX_LENGTH_NUMBER);
        for (int i = 0; i < length; i++) {
            int index = rand.getInt(alphabet.length() - 1);
            char randomChar = alphabet.charAt(index);
            sb.append(randomChar);
        }
        String randomString = sb.toString();
        this.text = randomString;
    }

    public String getQuestionInfo() {
        return text;
    }
}
