from django.shortcuts import redirect
from functools import wraps

def group_checks(group_names, redirect_url='core'):
    """
    Decorator to restrict access to users in specific groups, redirecting unauthorized users.

    Args:
        group_names (str or list): The group name(s) to check. Can be a string or a list of strings.
        redirect_url (str): The URL name to redirect unauthorized users. Defaults to 'core:index'.

    Returns:
        function: The wrapped view function with group access control.
    """
    if isinstance(group_names, str):
        group_names = [group_names]  # Convert a single group name to a list

    def inner(view_func):
        
        @wraps(view_func)
        def _wrapper(request, *args, **kwargs):

            # Check if the user belongs to at least one of the specified groups
            if not request.user.groups.filter(name__in=group_names).exists():
                return redirect(redirect_url)
            
            # Proceed with the view if the user is in the group
            return view_func(request, *args, **kwargs)
        
        return _wrapper
    
    return inner