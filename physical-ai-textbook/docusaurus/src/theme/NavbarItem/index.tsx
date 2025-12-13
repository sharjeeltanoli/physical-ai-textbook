import React, {type ReactNode} from 'react';
import NavbarItem from '@theme-original/NavbarItem';
import type NavbarItemType from '@theme/NavbarItem';
import type {WrapperProps} from '@docusaurus/types';

type Props = WrapperProps<typeof NavbarItemType>;

export default function NavbarItemWrapper(props: Props): ReactNode {
  return (
    <>
      <NavbarItem {...props} />
    </>
  );
}
