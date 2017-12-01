import csv


def filter(csvfile, include={}, exclude={}, fieldnames=None, delimiter=',', quotechar='"'):
    """
    Filters a csv file to include only items containing the correct values in the columns.

    Args:
        csv: The csv file to be filtered.
             Must be a list, file, or file-like object that supports the iterator protocol that returns strings as
             values.
             A column from the csv that contains strings will be considered a list delimited by the same delimiter as
             the columns themselves.
        include: A dictionary with a mapping from csv-header to a list or set of allowed values in the filtered csv.
                 In a column with lists as values, any 1 of the values from the list can match any of the expected
                 values from include.
        exclude: Works oppositely to include. If any value from the csv in the applicable header matches, exclude that
                 item from the returned list.
                 Note that exclude is stronger than include and will remove items that should otherwise have been
                 included.
        fieldnames: A list of headers, to be used if the csv does not have headers of their own.
                    If None, the first row of the csv is presumed to be the header columns.
        delimiter: The delimiter to be used to separate values in the csv.
        quotechar: The character which indicates the start and end of strings (lists).

    Returns:
        A list of strings from csvfile that passed the filters.

    Note:
        Any column that are not present in either include or exclude is going to be ignored and will be included
        blindly in the output list.

    Example usage:
        a_csv = ['a,b', '1,0', '"2,0",0', '5,5', '1,1']
        ret = filter(a_csv, include={'a': {'1', '0'}}, exclude={'b': {'1'}})
        # ret is equal to ['a,b, '1,0', '"2,0",0']
    """
    def filter_line(line, constraints):
        filters = []
        for constraint in constraints:
            constraint_line = line[constraint].split(',')
            constraint_line = [str(value) in constraints[constraint] for value in constraint_line]
            filters.append(any(constraint_line))
        return filters

    csvreader = csv.DictReader(csvfile, fieldnames=fieldnames, delimiter=delimiter, quotechar=quotechar)
    filtered_csv = []

    include_headers = not fieldnames
    fieldnames = csvreader.fieldnames

    if include_headers:
        linestring = ''
        for fn in fieldnames:
            if delimiter in fn:
                linestring += '{0}{1}{0}'.format(quotechar, fn) + delimiter
            else:
                linestring += fn + delimiter
        filtered_csv.append(linestring[:-len(delimiter)])

    for line in csvreader:
        if any(filter_line(line, exclude)):
            continue
        if sum(filter_line(line, include)) != len(include):
            continue

        linestring = ''
        for fn in fieldnames:
            if delimiter in line[fn]:
                linestring += '{0}{1}{0}'.format(quotechar, line[fn]) + delimiter
            else:
                linestring += line[fn] + delimiter
        filtered_csv.append(linestring[:-len(delimiter)])  # Remove the last delimiter from the line
    return filtered_csv
