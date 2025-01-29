def get_transitive(relations):
    new_transitive_set = set(relations)
    while True:
        new_relations = set((x, w) for x, y in new_transitive_set for q, w in new_transitive_set if q == y)

        temp_set = new_transitive_set | new_relations

        if temp_set == new_transitive_set:
            break

        new_transitive_set = temp_set

    return new_transitive_set


def get_reflexive(elements):
    new_reflexive_set = set()
    for i in elements:
        new_reflexive_set.add((i, i))
    return new_reflexive_set


relation = {("a", "b"), ("a", "c"), ("b", "d"), ("d", "e")}
set_of_elements = {"a", "b", "c", "d", "e"}

print("R=", get_transitive(relation) | get_reflexive(set_of_elements))
