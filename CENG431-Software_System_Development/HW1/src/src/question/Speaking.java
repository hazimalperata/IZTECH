package src.question;

import java.util.ArrayList;
import java.util.List;

import src.question.type.Audio;
import src.question.type.IType;

public class Speaking extends Question {
    List<IType> questionTypes;

    public Speaking() {
        this(8);
    }

    public Speaking(int points) {
        super(points);
        createQuestion();
    }

    private void createQuestion() {
        questionTypes = new ArrayList<IType>();
        IType audioQuestion1 = new Audio();
        IType audioQuestion2 = new Audio();
        questionTypes.add(audioQuestion1);
        questionTypes.add(audioQuestion2);

    }

    public List<IType> getQuestionTypes() {
        return questionTypes;
    }

    @Override
    public String toString() {
        return "S";
    }
}
